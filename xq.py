#!/usr/bin/env python3
"""A miller-style CLI tool for interacting with tabular data files."""

import os
import sys
import json
import logging
import hashlib
import pickle
import fnmatch
import traceback
import inspect
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Union
from functools import wraps

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
import typer
from typer import Option, Argument, Exit, Typer, Context
import re

# --- CLI Framework ---

@dataclass
class CommonOptions:
    """Common CLI options that can be used globally and locally."""
    format: Optional[str] = field(
        default=None,
        metadata={
            "option": ["--format", "-f"],
            "help": "Output format (table, json)",
            "default_tty": "table",
            "default_pipe": "json"
        }
    )
    verbose: int = field(
        default=1,
        metadata={
            "option": ["--verbose", "-v"],
            "help": "Verbosity (0-3)"
        }
    )



# Remove verbose request DTOs - use simple method parameters

# --- Output DTOs ---

@dataclass
class FieldInfo:
    """Information about a single field."""
    cid: int
    name: str
    short_name: str
    dtype: str

@dataclass
class SchemaResponse:
    """Response DTO for schema command."""
    fields: List[FieldInfo]
    computed_fields: Dict[str, str]
    file_name: str

@dataclass
class FilterResponse:
    """Response DTO for filter command."""
    records: List[Dict[str, Any]]

@dataclass
class TagInfo:
    """Information about a computed field tag."""
    name: str
    expression: str

@dataclass
class TagListResponse:
    """Response DTO for tag list command."""
    tags: List[TagInfo]

@dataclass
class TagSetResponse:
    """Response DTO for tag set command."""
    field_name: str
    success: bool
    message: str

@dataclass
class TagUnsetResponse:
    """Response DTO for tag unset command."""
    field_name: str
    success: bool
    message: str

class CLIError(Exception):
    """Base exception for CLI errors."""
    def __init__(self, message: str, exit_code: int = 1):
        self.message = message
        self.exit_code = exit_code
        super().__init__(message)

class ApplicationError(CLIError):
    """Exception for application logic errors."""
    pass

class ValidationError(CLIError):
    """Exception for input validation errors."""
    pass

class FileNotFoundError(CLIError):
    """Exception for file not found errors."""
    def __init__(self, file_path: str):
        super().__init__(f"File not found: {file_path}", exit_code=2)

class OutputRenderer:
    """Unified output renderer for all response DTOs."""

    def __init__(self, console: Console, format: Optional[str] = None, common_options: Optional[CommonOptions] = None):
        self.console = console
        self.format = self._determine_format(format, common_options)

    def _determine_format(self, format: Optional[str], common_options: Optional[CommonOptions]) -> str:
        """Determine output format using dataclass metadata for defaults."""
        if format:
            return format

        # Use metadata from CommonOptions for smart defaults
        if common_options:
            format_field = CommonOptions.__dataclass_fields__['format']
            if sys.stdout.isatty():
                return format_field.metadata.get('default_tty', 'table')
            else:
                return format_field.metadata.get('default_pipe', 'json')

        # Fallback
        return 'table' if sys.stdout.isatty() else 'json'

    def render(self, response: Any) -> None:
        """Render any response DTO based on its type."""
        if self.format == "json":
            self._render_json(response)
        else:
            self._render_table(response)

    def _render_json(self, response: Any) -> None:
        """Render response as JSON."""
        import json
        from dataclasses import asdict

        # For filter responses, output just the records array directly
        if isinstance(response, FilterResponse):
            self.console.print(json.dumps(response.records, indent=2))
        else:
            self.console.print(json.dumps(asdict(response), indent=2))

    def _render_table(self, response: Any) -> None:
        """Render response as a Rich table."""
        if isinstance(response, SchemaResponse):
            self._render_schema_table(response)
        elif isinstance(response, FilterResponse):
            self._render_filter_table(response)
        elif isinstance(response, TagListResponse):
            self._render_tag_list_table(response)
        elif isinstance(response, (TagSetResponse, TagUnsetResponse)):
            self._render_tag_operation_result(response)
        else:
            self.console.print(f"[yellow]Unknown response type: {type(response)}[/yellow]")

    def _render_schema_table(self, response: SchemaResponse) -> None:
        """Render schema response as table."""
        table = Table(title=f"Schema for {response.file_name}")
        table.add_column("CID", style="yellow")
        table.add_column("Field Name", style="cyan")
        table.add_column("Short Name", style="green")
        table.add_column("Data Type", style="magenta")

        for field in response.fields:
            table.add_row(str(field.cid), field.name, field.short_name, field.dtype)

        self.console.print(table)

        if response.computed_fields:
            computed_table = Table(title="Computed Fields (Tags)")
            computed_table.add_column("Tag Name", style="cyan")
            computed_table.add_column("Expression", style="green")
            for name, expr in response.computed_fields.items():
                computed_table.add_row(name, expr)
            self.console.print(computed_table)

    def _render_filter_table(self, response: FilterResponse) -> None:
        """Render filter response as table."""
        if not response.records:
            self.console.print("[yellow]No records found matching the query.[/yellow]")
            return

        # Get columns from first record
        columns = list(response.records[0].keys()) if response.records else []

        table = Table(title="Filtered Results", expand=True)
        for col in columns:
            table.add_column(col, style="cyan", no_wrap=True)

        for record in response.records:
            table.add_row(*[str(record.get(col, "")) for col in columns])

        self.console.print(table)

    def _render_tag_list_table(self, response: TagListResponse) -> None:
        """Render tag list response as table."""
        if not response.tags:
            self.console.print("[yellow]No computed fields (tags) found.[/yellow]")
            return

        table = Table(title="Computed Fields (Tags)")
        table.add_column("Tag Name", style="cyan")
        table.add_column("Expression", style="green")

        for tag in response.tags:
            table.add_row(tag.name, tag.expression)

        self.console.print(table)

    def _render_tag_operation_result(self, response: Union[TagSetResponse, TagUnsetResponse]) -> None:
        """Render tag operation result."""
        if response.success:
            self.console.print(f"[green]{response.message}[/green]")
        else:
            self.console.print(f"[red]{response.message}[/red]")

class TyperCommonOptions:
    """
    Generic Typer wrapper with:
    - Automatic common option injection from dataclass
    - Exception handling with full stack traces
    - Clean method routing via decorators
    """

    def __init__(self, app_title: str, app_class: type, common_options_class: type = CommonOptions, needs_file_input: bool = True):
        self.console = Console()
        self.app = typer.Typer(help=app_title)
        self.app_class = app_class
        self.app_instance = None
        self.common_options_class = common_options_class
        self.global_options = common_options_class()
        self.needs_file_input = needs_file_input

        # Parse positional file argument if needed
        if self.needs_file_input:
            self.file_path = self._extract_file_arg()

        # Setup global callback with dynamic options
        self._setup_global_callback()

    def _setup_global_callback(self):
        """Setup global callback with options derived from CommonOptions dataclass."""
        # Create callback signature from CommonOptions fields
        params = [inspect.Parameter('ctx', inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=Context)]

        for field_name, field in self.common_options_class.__dataclass_fields__.items():
            option_args = field.metadata.get('option', [f'--{field_name}'])
            help_text = field.metadata.get('help', '')
            default_val = field.default if field.default != field.default_factory else field.default_factory()

            param = inspect.Parameter(
                field_name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=Option(default_val, *option_args, help=help_text),
                annotation=field.type
            )
            params.append(param)

        def callback_impl(ctx: Context, **kwargs):
            # Update global options
            for key, value in kwargs.items():
                if hasattr(self.global_options, key):
                    setattr(self.global_options, key, value)

            # Setup logging
            if hasattr(self.global_options, 'verbose'):
                self._setup_logging(self.global_options.verbose)

        # Create function with dynamic signature
        callback_impl.__signature__ = inspect.Signature(params)
        self.app.callback(invoke_without_command=True)(callback_impl)

    def _setup_logging(self, verbose: int):
        """Setup logging based on verbosity level."""
        level_map = {
            0: logging.ERROR,
            1: logging.WARNING,
            2: logging.INFO,
            3: logging.DEBUG
        }
        level = level_map.get(verbose, logging.DEBUG)
        logging.basicConfig(
            level=level,
            stream=sys.stderr,
            format='%(levelname)s: %(message)s'
        )

    def _extract_file_arg(self) -> Optional[str]:
        """Extract file argument from command line - simple positional before any commands."""
        args = sys.argv[1:]

        for i, arg in enumerate(args):
            if not arg.startswith('-') and ('.' in arg or '/' in arg or os.path.exists(arg)):
                # Remove file arg from sys.argv so Typer doesn't see it
                sys.argv.pop(i + 1)  # +1 because sys.argv includes script name
                return arg
        return None

    def _get_app_instance(self) -> Any:
        """Get or create application instance with file loaded and options applied generically."""
        if self.app_instance is None:
            self.app_instance = self.app_class()

            # Load file if needed - generic file loading
            if self.needs_file_input and hasattr(self, 'file_path') and self.file_path:
                if hasattr(self.app_instance, 'load_data'):
                    self.app_instance.load_data(self.file_path)

        # Apply common options generically using reflection
        for field_name, field in self.common_options_class.__dataclass_fields__.items():
            option_value = getattr(self.global_options, field_name)
            setter_method = f'set_{field_name}'

            if hasattr(self.app_instance, setter_method):
                getattr(self.app_instance, setter_method)(option_value)

        return self.app_instance

    def _handle_exceptions(self, func: Callable) -> Callable:
        """Decorator to handle exceptions - ALWAYS show full stacks for developers."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except CLIError as e:
                # Show the error message but also the full stack trace
                self.console.print(f"[bold red]CLI Error:[/bold red] {e.message}")
                self.console.print(Panel(
                    Text(traceback.format_exc(), style="red"),
                    title="[bold red]Full Stack Trace[/bold red]",
                    border_style="red"
                ))
                raise Exit(code=e.exit_code)
            except Exception as e:
                # ALWAYS show full traceback for developers
                self.console.print(Panel(
                    Text(traceback.format_exc(), style="red"),
                    title="[bold red]Exception Stack Trace[/bold red]",
                    border_style="red"
                ))
                raise Exit(code=1)
        return wrapper

    def command(self, name: str, method_name: str = None):
        """Register a command that calls an app method and renders response."""
        def decorator(func: Callable):
            @self._handle_exceptions
            def wrapper(**kwargs):
                app_instance = self._get_app_instance()

                # Extract common option names
                common_option_names = set(self.common_options_class.__dataclass_fields__.keys())

                # Split kwargs into common options and method args
                method_kwargs = {k: v for k, v in kwargs.items() if k not in common_option_names}
                common_kwargs = {k: v for k, v in kwargs.items() if k in common_option_names}

                # Call the app method
                app_method_name = method_name or name
                if hasattr(app_instance, app_method_name):
                    app_method = getattr(app_instance, app_method_name)
                    response = app_method(**method_kwargs)
                else:
                    raise ValueError(f"Application has no method '{app_method_name}'")

                # Render response if present using smart format detection
                if response is not None:
                    # Create options object for renderer from command-level options
                    render_options = self.common_options_class()
                    for key, value in common_kwargs.items():
                        setattr(render_options, key, value)

                    # Use command-level format if provided, otherwise fall back to global
                    effective_format = render_options.format or self.global_options.format
                    renderer = OutputRenderer(self.console, effective_format, render_options)
                    renderer.render(response)

                return response

            # Create clean signature for Typer (only non-common options)
            orig_sig = inspect.signature(func)
            clean_params = []

            for param_name, param in orig_sig.parameters.items():
                if param_name not in ['app'] and param_name not in self.common_options_class.__dataclass_fields__:
                    clean_params.append(param)

            # Add common options to signature
            for field_name, field in self.common_options_class.__dataclass_fields__.items():
                option_args = field.metadata.get('option', [f'--{field_name}'])
                help_text = field.metadata.get('help', '')
                default_val = field.default if field.default != field.default_factory else field.default_factory()

                param = inspect.Parameter(
                    field_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=Option(default_val, *option_args, help=help_text),
                    annotation=field.type
                )
                clean_params.append(param)

            wrapper.__signature__ = inspect.Signature(clean_params)
            wrapper.__name__ = func.__name__
            wrapper.__doc__ = func.__doc__

            return self.app.command(name)(wrapper)
        return decorator

    def subcommand_group(self, name: str, help_text: str = None) -> 'TyperCommonOptions':
        """Create a subcommand group."""
        subapp = typer.Typer(help=help_text or f"{name} commands")
        self.app.add_typer(subapp, name=name)

        # Create a new wrapper for the subcommand group sharing state
        sub_wrapper = TyperCommonOptions.__new__(TyperCommonOptions)
        sub_wrapper.console = self.console
        sub_wrapper.app = subapp
        sub_wrapper.app_class = self.app_class
        sub_wrapper.app_instance = self.app_instance  # Share instance
        sub_wrapper.needs_file_input = self.needs_file_input
        sub_wrapper.common_options_class = self.common_options_class
        sub_wrapper.global_options = self.global_options  # Share global state
        if hasattr(self, 'file_path'):
            sub_wrapper.file_path = self.file_path
        return sub_wrapper

    def run(self):
        """Run the Typer application."""
        self.app()


# --- Abbreviation Logic (Internal) ---

def abbreviate_internal(text: str, max_length: int) -> str:
    """A simple internal function to abbreviate text."""
    if len(text) <= max_length:
        return text

    words = text.split()
    abbr = ""
    for word in words:
        if len(abbr) + len(word) + (1 if abbr else 0) > max_length:
            break
        abbr += (" " if abbr else "") + word
    return abbr if abbr else text[:max_length]


# --- Configuration & Application State ---

APP_NAME = "xq"
CONFIG_DIR = Path.home() / ".config" / APP_NAME
CACHE_DIR = Path.home() / ".cache" / APP_NAME

@dataclass
class DataAppConfig:
    """Data application configuration."""
    output_format: str = "table"
    verbose: int = 1

@dataclass
class FieldAlias:
    """Represents the alias information for a single field."""
    cid: int
    short_name: str
    dtype: str

@dataclass
class FileConfig:
    """Represents the configuration for a single data file."""
    version: int = 1
    fields: Dict[str, FieldAlias] = field(default_factory=dict)
    computed: Dict[str, str] = field(default_factory=dict)


@dataclass
class AppState:
    """Application state."""
    config: DataAppConfig = field(default_factory=DataAppConfig)
    df: Optional[pd.DataFrame] = None
    file_path: Optional[Path] = None
    content_hash: Optional[str] = None
    file_config: FileConfig = field(default_factory=FileConfig)
    short_name_map: Dict[str, str] = field(default_factory=dict)


class DataApplication:
    """Main application class with business logic separated from CLI concerns."""

    def __init__(self):
        self.state = AppState()
        self.console = Console()
        self._setup()

    def _setup(self):
        """Initial setup for config and cache directories."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def set_output_format(self, format: str):
        """Set the output format."""
        self.state.config.output_format = format

    def set_verbosity(self, verbose: int):
        """Set the verbosity level."""
        self.state.config.verbose = verbose

    def get_content_hash(self, file_path: Path) -> str:
        """Computes the SHA1 hash of the file's content."""
        h = hashlib.sha1()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()

    def get_cache_path(self, content_hash: str) -> Path:
        """Generates a cache path for a given content hash."""
        return CACHE_DIR / f"{content_hash}.pkl"

    def get_config_path(self, content_hash: str) -> Path:
        """Generates a config path for a given content hash."""
        return CONFIG_DIR / f"{content_hash}.json"

    def save_file_config(self):
        """Saves the file-specific configuration (fields, computed, etc.)."""
        if not self.state.content_hash:
            logging.warning("Cannot save file config, content hash not available.")
            return

        config_path = self.get_config_path(self.state.content_hash)

        # Convert dataclasses to dicts for JSON serialization
        config_dict = {
            "version": self.state.file_config.version,
            "fields": {k: v.__dict__ for k, v in self.state.file_config.fields.items()},
            "computed": self.state.file_config.computed
        }

        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        logging.info(f"Saved file config to {config_path}")

    def _generate_aliases(self, df: pd.DataFrame, base_aliases: Optional[Dict[str, FieldAlias]] = None):
        """Generates and stores unique short names for fields."""
        aliases: Dict[str, FieldAlias] = base_aliases if base_aliases is not None else {}
        used_short_names = {alias.short_name for alias in aliases.values()}

        max_cid = -1
        if aliases:
            max_cid = max(alias.cid for alias in aliases.values())

        for i, col_name in enumerate(df.columns):
            if col_name in aliases:
                continue # Skip columns that already have an alias

            # Generate a base short name using the internal abbreviate function
            base_name = abbreviate_internal(col_name, max_length=20)
            base_name = re.sub(r'[^a-z0-9]+', '-', base_name.lower().strip()).strip('-')

            short_name = base_name
            counter = 2
            while short_name in used_short_names:
                short_name = f"{base_name}-{counter}"
                counter += 1

            used_short_names.add(short_name)
            max_cid += 1
            aliases[col_name] = FieldAlias(
                cid=max_cid,
                short_name=short_name,
                dtype=str(df[col_name].dtype)
            )

        self.state.file_config.fields = aliases
        self.save_file_config()

    def _apply_computed_fields(self):
        """Applies computed field expressions to the DataFrame in dependency order."""
        if self.state.df is None or not self.state.file_config.computed:
            return

        computed_fields = self.state.file_config.computed
        computed_names = set(computed_fields.keys())

        # Build dependency graph for topological sort
        adj = {name: [] for name in computed_names}
        in_degree = {name: 0 for name in computed_names}

        for name, expr in computed_fields.items():
            deps_in_expr = set(re.findall(r'\{([^{}]+)\}', expr))
            for dep_name in deps_in_expr:
                if dep_name in computed_names:
                    # `dep_name` is a dependency of `name`
                    adj[dep_name].append(name)
                    in_degree[name] += 1

        # Kahn's algorithm for topological sort
        queue = [name for name in computed_names if in_degree[name] == 0]
        eval_order = []
        while queue:
            u = queue.pop(0)
            eval_order.append(u)
            for v in adj[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)

        if len(eval_order) < len(computed_names):
            cycle_nodes = sorted(list(computed_names - set(eval_order)))
            raise ApplicationError(f"Circular dependency detected in computed fields: {', '.join(cycle_nodes)}")

        # Evaluate expressions in order
        for name in eval_order:
            expression = computed_fields[name]
            try:
                translated_expression = self._translate_expression(expression)
                self.state.df[name] = self.state.df.eval(translated_expression, engine='python')
                logging.info(f"Applied computed field '{name}' with expression: {expression}")
            except Exception as e:
                logging.warning(f"Could not compute field '{name}': {e}")
                if name in self.state.df.columns:
                    del self.state.df[name]

        # After adding all computed columns, update aliases for them
        self._generate_aliases(self.state.df, base_aliases=self.state.file_config.fields)

    def _translate_expression(self, expression: str) -> str:
        """
        Translates an expression from {short_name} or {computed_field_name} syntax
        to a pandas-compatible backticked expression.
        Raises ValueError if a name is not found.
        """
        translated_expression = expression

        # Find all {name} patterns, sorting by length descending to avoid substring issues
        names_in_expr = sorted(list(set(re.findall(r'\{([^{}]+)\}', expression))), key=len, reverse=True)

        for name in names_in_expr:
            if name in self.state.short_name_map:
                full_name = self.state.short_name_map[name]
                # Using backticks to handle column names with spaces or special characters
                translated_expression = translated_expression.replace(f'{{{name}}}', f'`{full_name}`')
            elif name in self.state.file_config.computed:
                # Computed fields are referenced by their name, which becomes a column name
                translated_expression = translated_expression.replace(f'{{{name}}}', f'`{name}`')
            else:
                raise ValidationError(f"Field name or alias '{{{name}}}' not found.")

        return translated_expression

    def load_data(self, file_path_str: str):
        """Loads data from a file, using cache if available."""
        file_path = Path(file_path_str)
        if not file_path.exists():
            raise FileNotFoundError(file_path_str)

        self.state.file_path = file_path
        self.state.content_hash = self.get_content_hash(file_path)

        cache_path = self.get_cache_path(self.state.content_hash)
        config_path = self.get_config_path(self.state.content_hash)

        # Load from cache if it exists
        if cache_path.exists():
            logging.info(f"Loading cached data from {cache_path}")
            with open(cache_path, 'rb') as f:
                self.state.df = pickle.load(f)
        else:
            logging.info(f"Loading data from {file_path}")
            file_ext = file_path.suffix.lower()
            try:
                if file_ext == '.xlsx':
                    self.state.df = pd.read_excel(file_path)
                elif file_ext == '.json':
                    self.state.df = pd.read_json(file_path)
                elif file_ext == '.csv':
                    self.state.df = pd.read_csv(file_path)
                else:
                    raise ApplicationError(f"Unsupported file type: {file_ext}")

                with open(cache_path, 'wb') as f:
                    pickle.dump(self.state.df, f)
                logging.info(f"Cached data to {cache_path}")
            except Exception as e:
                raise ApplicationError(f"Error loading file: {e}")

        # Load or generate aliases
        if config_path.exists():
            logging.info(f"Loading field aliases from {config_path}")
            with open(config_path, 'r') as f:
                config_data = json.load(f)

            # Migration from v0 to v1
            if "version" not in config_data:
                logging.info("Old config format (v0) detected. Migrating to v1.")
                migrated_fields = {k: FieldAlias(**v) for k, v in config_data.items()}
                self.state.file_config = FileConfig(version=1, fields=migrated_fields, computed={})
                self.save_file_config()
            else:
                self.state.file_config = FileConfig(
                    version=config_data.get("version", 1),
                    fields={k: FieldAlias(**v) for k, v in config_data.get("fields", {}).items()},
                    computed=config_data.get("computed", {})
                )
        else:
            logging.info("Generating new field aliases.")
            self._generate_aliases(self.state.df)

        # Create a reverse map for expression translation and filtering.
        self.state.short_name_map = {v.short_name: k for k, v in self.state.file_config.fields.items()}

        # Apply computed fields
        self._apply_computed_fields()

        # Re-create the map in case computed fields added new aliases.
        self.state.short_name_map = {v.short_name: k for k, v in self.state.file_config.fields.items()}

    def _match_field_pattern(self, pattern: str) -> List[str]:
        """
        Matches field names and aliases using wildcard patterns (* and ?).
        Returns a list of actual column names that match the pattern.
        """
        if self.state.df is None:
            return []

        matched_columns = []
        all_columns = list(self.state.df.columns)
        all_aliases = list(self.state.short_name_map.keys())

        # Check if pattern matches any column names directly
        for col in all_columns:
            if fnmatch.fnmatch(col, pattern):
                matched_columns.append(col)

        # Check if pattern matches any aliases, and resolve to column names
        for alias in all_aliases:
            if fnmatch.fnmatch(alias, pattern):
                resolved_field = self.state.short_name_map[alias]
                if resolved_field not in matched_columns:
                    matched_columns.append(resolved_field)

        return matched_columns

    def get_schema(self) -> Optional[Dict[str, FieldAlias]]:
        """Gets the schema of the loaded DataFrame."""
        if self.state.df is None:
            raise ApplicationError("No data loaded.")
        return self.state.file_config.fields

    def filter_data(self, query: str) -> Optional[pd.DataFrame]:
        """
        Filters data based on a comma-separated query string.
        Returns a DataFrame with columns corresponding to the query fields.
        e.g. 'first-name/^A/,last-name'
        """
        if self.state.df is None:
            raise ApplicationError("No data loaded.")

        display_columns = []
        filters = []
        has_wildcard = False
        expressions = [expr.strip() for expr in query.split(',')]
        # More specific operators first to avoid partial matches (e.g., <= before <)
        operators = ['~', '>=', '<=', '>', '<', '=']

        try:
            for expression in expressions:
                if expression == '*':
                    has_wildcard = True
                    continue

                field_name_or_alias = expression
                operator = None
                value = None

                # Shorthand regex format: 'field/regex/'
                if '/' in expression and '~' not in expression and '=' not in expression and '>' not in expression and '<' not in expression:
                    parts = expression.split('/', 1)
                    field_name_or_alias = parts[0]
                    operator = '~'
                    value = parts[1]
                    if value.endswith('/'):
                        value = value[:-1]
                else:
                    # Find which operator is in the expression
                    for op in operators:
                        if op in expression:
                            parts = expression.split(op, 1)
                            if len(parts) == 2 and parts[0]:
                                field_name_or_alias = parts[0]
                                operator = op
                                value = parts[1]
                                break

                # Check if field name contains wildcards
                if '*' in field_name_or_alias or '?' in field_name_or_alias:
                    # Handle wildcard patterns
                    matched_fields = self._match_field_pattern(field_name_or_alias)
                    if not matched_fields:
                        raise ValidationError(f"No fields match pattern: {field_name_or_alias}")

                    # Add all matched fields to display columns
                    for matched_field in matched_fields:
                        if matched_field not in display_columns:
                            display_columns.append(matched_field)

                    # If there's an operator, apply it to all matched fields
                    if operator:
                        for matched_field in matched_fields:
                            filters.append((matched_field, operator.strip(), value.strip()))
                else:
                    # Handle exact field name or alias
                    resolved_field = self.state.short_name_map.get(field_name_or_alias, field_name_or_alias)

                    if resolved_field not in self.state.df.columns:
                        raise ValidationError(f"Invalid field name or alias: {field_name_or_alias}")

                    if resolved_field not in display_columns:
                        display_columns.append(resolved_field)

                    if operator:
                        filters.append((resolved_field, operator.strip(), value.strip()))

            if has_wildcard:
                all_columns = list(self.state.df.columns)
                # Add remaining columns that aren't already there
                for col in all_columns:
                    if col not in display_columns:
                        display_columns.append(col)

            # Apply all filters sequentially (AND logic)
            filtered_df = self.state.df.copy()
            for field, op, val in filters:
                if op == '~':
                    if pd.api.types.is_string_dtype(filtered_df[field]):
                        mask = filtered_df[field].str.contains(val, regex=True, na=False)
                        filtered_df = filtered_df[mask]
                    else:
                        raise ValidationError(f"Regex filter '~' can only be applied to string columns. '{field}' is not a string type.")
                elif op in ['>=', '<=', '>', '<', '=']:
                    # Attempt to convert value to number for comparison
                    try:
                        numeric_val = float(val)
                    except ValueError:
                        # Fallback to string comparison for '=' operator
                        if op == '=':
                            mask = filtered_df[field].astype(str) == val
                            filtered_df = filtered_df[mask]
                            continue
                        raise ValidationError(f"Operator '{op}' requires a numeric value for comparison (e.g., 'score>5').")

                    if pd.api.types.is_numeric_dtype(filtered_df[field]):
                        col = pd.to_numeric(filtered_df[field])
                        if op == '=': mask = col == numeric_val
                        elif op == '>': mask = col > numeric_val
                        elif op == '<': mask = col < numeric_val
                        elif op == '>=': mask = col >= numeric_val
                        elif op == '<=': mask = col <= numeric_val
                        filtered_df = filtered_df[mask]
                    else:
                        raise ValidationError(f"Cannot apply numeric comparison '{op}' on non-numeric column '{field}'.")
                else:
                    raise ValidationError(f"Unsupported operator: {op}")

            # Return the filtered DataFrame with only the specified columns
            return filtered_df[display_columns]

        except re.error as e:
            raise ValidationError(f"Invalid regex pattern: {e}")

    # --- Business Logic Methods ---

    def show_schema(self) -> SchemaResponse:
        """Get the schema of the data file."""
        schema_data = self.get_schema()
        if not schema_data:
            raise ApplicationError("No schema data available")

        # Sort by cid for consistent order
        sorted_aliases = sorted(schema_data.items(), key=lambda item: item[1].cid)

        fields = [
            FieldInfo(
                cid=alias.cid,
                name=name,
                short_name=alias.short_name,
                dtype=alias.dtype
            )
            for name, alias in sorted_aliases
        ]

        return SchemaResponse(
            fields=fields,
            computed_fields=self.state.file_config.computed,
            file_name=self.state.file_path.name
        )

    def filter_records(self, query: str) -> FilterResponse:
        """Filter records and return the results."""
        result_df = self.filter_data(query)
        if result_df is None:
            raise ApplicationError("Failed to filter data")

        records = [dict(row) for _, row in result_df.iterrows()]

        return FilterResponse(records=records)

    def list_tags(self) -> TagListResponse:
        """List all computed fields (tags)."""
        computed_fields = self.state.file_config.computed

        tags = [
            TagInfo(name=name, expression=expr)
            for name, expr in computed_fields.items()
        ]

        return TagListResponse(tags=tags)

    def set_tag(self, field_name: str, expression: str) -> TagSetResponse:
        """Set a computed field (tag) with a pandas expression."""
        try:
            # Validate expression by trying to apply it
            translated_expression = self._translate_expression(expression)
            self.state.df.eval(translated_expression, engine='python')

            self.state.file_config.computed[field_name] = expression
            self.save_file_config()

            return TagSetResponse(
                field_name=field_name,
                success=True,
                message=f"Tag '{field_name}' set successfully."
            )
        except Exception as e:
            return TagSetResponse(
                field_name=field_name,
                success=False,
                message=f"Failed to set tag '{field_name}': {e}"
            )

    def unset_tag(self, field_name: str) -> TagUnsetResponse:
        """Remove a computed field (tag)."""
        if field_name in self.state.file_config.computed:
            del self.state.file_config.computed[field_name]

            # also remove from fields if it exists
            if field_name in self.state.file_config.fields:
                del self.state.file_config.fields[field_name]

            self.save_file_config()
            return TagUnsetResponse(
                field_name=field_name,
                success=True,
                message=f"Tag '{field_name}' unset successfully."
            )
        else:
            return TagUnsetResponse(
                field_name=field_name,
                success=False,
                message=f"Tag '{field_name}' not found."
            )


# --- CLI Command Handlers ---

def create_cli():
    """Create and configure the CLI application."""
    cli = TyperCommonOptions("A CLI for querying tabular data.", DataApplication)

    # Main commands - proper method calls
    @cli.command("schema", "show_schema")
    def schema_cmd(app: DataApplication):
        """Display the schema of the data file."""
        return app.show_schema()

    @cli.command("flt", "filter_records")
    def flt_cmd(
        app: DataApplication,
        query: str = Argument(..., help="Filter query, comma-separated (e.g., 'first-name/^A/,score>5')"),
    ):
        """Filter records and display specified columns."""
        return app.filter_records(query)

    # Tag subcommands
    tag_cli = cli.subcommand_group("tag", "Commands for managing computed fields (tags)")

    @tag_cli.command("ls", "list_tags")
    def tag_ls_cmd(app: DataApplication):
        """List all computed fields (tags)."""
        return app.list_tags()

    @tag_cli.command("set", "set_tag")
    def tag_set_cmd(
        app: DataApplication,
        field_name: str = Argument(..., help="The name of the new computed field."),
        expression: str = Argument(..., help="The pandas expression to compute the field."),
    ):
        """Set a computed field (tag) with a pandas expression."""
        return app.set_tag(field_name, expression)

    @tag_cli.command("unset", "unset_tag")
    def tag_unset_cmd(
        app: DataApplication,
        field_name: str = Argument(..., help="The name of the computed field to remove."),
    ):
        """Remove a computed field (tag)."""
        return app.unset_tag(field_name)

    return cli


if __name__ == "__main__":
    cli = create_cli()
    cli.run()