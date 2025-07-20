#!/usr/bin/env python3
"""A miller-style CLI tool for interacting with tabular data files."""

import os
import sys
import json
import logging
import hashlib
import pickle
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import pandas as pd
from rich.console import Console
from rich.table import Table
from typer_di import TyperDI, Depends
from typer import Option, Argument, Exit, Typer
import re

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
class AppConfig:
    """Application configuration."""
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
    config: AppConfig = field(default_factory=AppConfig)
    df: Optional[pd.DataFrame] = None
    file_path: Optional[Path] = None
    content_hash: Optional[str] = None
    file_config: FileConfig = field(default_factory=FileConfig)
    short_name_map: Dict[str, str] = field(default_factory=dict)


class Application:
    """Main application class."""

    def __init__(self, typer_app: TyperDI):
        self.typer = typer_app
        self.state = AppState()
        self.console = Console()
        self._setup()

    def _setup(self):
        """Initial setup for config and cache directories."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._setup_logging()

    def _setup_logging(self):
        """Sets up logging based on verbosity."""
        level = logging.WARNING
        if self.state.config.verbose == 0:
            level = logging.ERROR
        elif self.state.config.verbose == 2:
            level = logging.INFO
        elif self.state.config.verbose >= 3:
            level = logging.DEBUG

        logging.basicConfig(level=level, stream=sys.stderr, format='%(levelname)s: %(message)s')

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
            self.console.print("[bold red]Error:[/bold red] Circular dependency detected in computed fields.")
            self.console.print(f"[bold red]Problematic fields:[/bold red] {', '.join(cycle_nodes)}")
            raise Exit(code=1)

        # Evaluate expressions in order
        for name in eval_order:
            expression = computed_fields[name]
            try:
                translated_expression = self._translate_expression(expression)
                self.state.df[name] = self.state.df.eval(translated_expression, engine='python')
                logging.info(f"Applied computed field '{name}' with expression: {expression}")
            except Exception as e:
                self.console.print(f"[bold yellow]Warning:[/bold yellow] Could not compute field '{name}': {e}")
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
                raise ValueError(f"Field name or alias '{{{name}}}' not found.")

        return translated_expression


    def load_data(self, file_path_str: str):
        """Loads data from a file, using cache if available."""
        file_path = Path(file_path_str)
        if not file_path.exists():
            self.console.print(f"[bold red]Error:[/bold red] File not found at {file_path_str}")
            raise Exit(code=1)

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
                    raise ValueError(f"Unsupported file type: {file_ext}")

                with open(cache_path, 'wb') as f:
                    pickle.dump(self.state.df, f)
                logging.info(f"Cached data to {cache_path}")
            except Exception as e:
                self.console.print(f"[bold red]Error loading file:[/bold red] {e}")
                raise Exit(code=1)

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


    def get_schema(self) -> Optional[Dict[str, FieldAlias]]:
        """Gets the schema of the loaded DataFrame."""
        if self.state.df is None:
            self.console.print("[bold red]No data loaded.[/bold red]")
            return None
        return self.state.file_config.fields

    def filter_data(self, query: str) -> Optional[pd.DataFrame]:
        """
        Filters data based on a comma-separated query string.
        Returns a DataFrame with columns corresponding to the query fields.
        e.g. 'first-name/^A/,last-name'
        """
        if self.state.df is None:
            self.console.print("[bold red]No data loaded.[/bold red]")
            return None

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

                # Resolve alias to original field name
                resolved_field = self.state.short_name_map.get(field_name_or_alias, field_name_or_alias)

                if resolved_field not in self.state.df.columns:
                    self.console.print(f"[bold red]Invalid field name or alias:[/bold red] {field_name_or_alias}")
                    return None

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
                        self.console.print(f"[bold red]Error:[/bold red] Regex filter '~' can only be applied to string columns. '{field}' is not a string type.")
                        return None
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
                        self.console.print(f"[bold red]Error:[/bold red] Operator '{op}' requires a numeric value for comparison (e.g., 'score>5').")
                        return None

                    if pd.api.types.is_numeric_dtype(filtered_df[field]):
                        col = pd.to_numeric(filtered_df[field])
                        if op == '=': mask = col == numeric_val
                        elif op == '>': mask = col > numeric_val
                        elif op == '<': mask = col < numeric_val
                        elif op == '>=': mask = col >= numeric_val
                        elif op == '<=': mask = col <= numeric_val
                        filtered_df = filtered_df[mask]
                    else:
                        self.console.print(f"[bold red]Error:[/bold red] Cannot apply numeric comparison '{op}' on non-numeric column '{field}'.")
                        return None
                else:
                    self.console.print(f"[bold red]Unsupported operator:[/bold red] {op}")
                    return None

            # Return the filtered DataFrame with only the specified columns
            return filtered_df[display_columns]

        except ValueError as e:
            self.console.print(f"[bold red]Invalid query:[/bold red] {e}")
            return None
        except re.error as e:
            self.console.print(f"[bold red]Invalid regex pattern:[/bold red] {e}")
            return None

# --- CLI ---

app = Application(
    TyperDI(
        help="A miller-style CLI for querying tabular data.",
        context_settings={"help_option_names": ["--help", "-h"], "allow_interspersed_args": False}
    )
)
# No callback. Typer will handle the help message.

def common_options(
    format: Optional[str] = Option(None, "--format", "-f", help="Output format (table, json)."),
    verbose: int = Option(1, "--verbose", "-v", help="Verbosity (0-3)."),
):
    """Dependency for common options."""
    app.state.config.verbose = verbose
    app._setup_logging()

    if format:
        app.state.config.output_format = format
    elif not sys.stdout.isatty():
        app.state.config.output_format = "json"

    return app


@app.typer.command("schema")
def schema(
    app_instance: Application = Depends(common_options)
):
    """Display the schema of the data file."""
    if app_instance.state.df is None:
        app.console.print("[bold red]Error:[/bold red] File path must be provided before the command.")
        raise Exit(code=1)

    schema_data = app_instance.get_schema()
    if schema_data:
        # Sort by cid for consistent order
        sorted_aliases = sorted(schema_data.items(), key=lambda item: item[1].cid)

        if app_instance.state.config.output_format == 'json':
            # Create a JSON-friendly representation
            json_output = {
                "fields": {
                    name: {
                        "cid": alias.cid,
                        "short_name": alias.short_name,
                        "dtype": alias.dtype
                    } for name, alias in sorted_aliases
                },
                "computed": app.state.file_config.computed
            }
            app_instance.console.print_json(data=json_output)
        else:
            table = Table(title=f"Schema for {app_instance.state.file_path.name}")
            table.add_column("CID", style="yellow")
            table.add_column("Field Name", style="cyan")
            table.add_column("Short Name", style="green")
            table.add_column("Data Type", style="magenta")
            for name, alias in sorted_aliases:
                table.add_row(str(alias.cid), name, alias.short_name, alias.dtype)
            app_instance.console.print(table)

            if app_instance.state.file_config.computed:
                computed_table = Table(title="Computed Fields (Tags)")
                computed_table.add_column("Tag Name", style="cyan")
                computed_table.add_column("Expression", style="green")
                for name, expr in app_instance.state.file_config.computed.items():
                    computed_table.add_row(name, expr)
                app_instance.console.print(computed_table)

@app.typer.command("flt")
def flt(
    query: str = Argument(..., help="Filter query, comma-separated (e.g., 'first-name/^A/,score>5')"),
    app_instance: Application = Depends(common_options)
):
    """Filter records and display specified columns."""
    if app_instance.state.df is None:
        app.console.print("[bold red]Error:[/bold red] File path must be provided before the command.")
        raise Exit(code=1)

    result_df = app_instance.filter_data(query)
    # logging.debug(f"Filtered DataFrame:\n{result_df}")
    if result_df is not None:
        if app_instance.state.config.output_format == 'json':
            app_instance.console.print(result_df.to_json(orient='records', indent=2))
        else:
            if result_df.empty:
                app_instance.console.print("[yellow]No records found matching the query.[/yellow]")
                return

            # print(result_df) # Temporary print for debugging

            table = Table(title="Filtered Results", expand=True)
            for col in result_df.columns:
                table.add_column(col, style="cyan", no_wrap=True)
            for _, row in result_df.iterrows():
                table.add_row(*[str(item) for item in row])
            app_instance.console.print(table)


@app.typer.command("tag-ls")
def tag_ls(
    app_instance: Application = Depends(common_options)
):
    """List all computed fields (tags)."""
    if app_instance.state.df is None:
        app.console.print("[bold red]Error:[/bold red] File path must be provided before the command.")
        raise Exit(code=1)

    computed_fields = app_instance.state.file_config.computed
    if not computed_fields:
        app_instance.console.print("[yellow]No computed fields (tags) found.[/yellow]")
        return

    table = Table(title="Computed Fields (Tags)")
    table.add_column("Tag Name", style="cyan")
    table.add_column("Expression", style="green")

    for name, expr in computed_fields.items():
        table.add_row(name, expr)

    app_instance.console.print(table)


@app.typer.command("tag-set")
def tag_set(
    field_name: str = Argument(..., help="The name of the new computed field."),
    expression: str = Argument(..., help="The pandas expression to compute the field."),
    app_instance: Application = Depends(common_options)
):
    """Set a computed field (tag) with a pandas expression."""
    if app_instance.state.df is None:
        app.console.print("[bold red]Error:[/bold red] File path must be provided before the command.")
        raise Exit(code=1)

    # Validate expression by trying to apply it
    try:
        translated_expression = app_instance._translate_expression(expression)
        app_instance.state.df.eval(translated_expression, engine='python')
    except Exception as e:
        app.console.print(f"[bold red]Invalid expression:[/bold red] {e}")
        raise Exit(code=1)

    app_instance.state.file_config.computed[field_name] = expression
    app_instance.save_file_config()
    app.console.print(f"[green]Tag '{field_name}' set successfully.[/green]")


@app.typer.command("tag-unset")
def tag_unset(
    field_name: str = Argument(..., help="The name of the computed field to remove."),
    app_instance: Application = Depends(common_options)
):
    """Remove a computed field (tag)."""
    if app_instance.state.df is None:
        app.console.print("[bold red]Error:[/bold red] File path must be provided before the command.")
        raise Exit(code=1)

    if field_name in app_instance.state.file_config.computed:
        del app_instance.state.file_config.computed[field_name]

        # also remove from fields if it exists
        if field_name in app_instance.state.file_config.fields:
            del app_instance.state.file_config.fields[field_name]

        app_instance.save_file_config()
        app.console.print(f"[green]Tag '{field_name}' unset successfully.[/green]")
    else:
        app.console.print(f"[bold yellow]Warning:[/bold yellow] Tag '{field_name}' not found.")
        raise Exit(code=2)


if __name__ == "__main__":
    COMMANDS = {"schema", "flt", "tag-set", "tag-unset", "tag-ls", "--help", "-h"}

    # Pre-process sys.argv to "shift" the file argument out
    if len(sys.argv) > 1 and sys.argv[1] not in COMMANDS:
        file_path = sys.argv.pop(1)
        # The app.load_data method will handle errors and exit if the file is invalid
        app.load_data(file_path)

    app.typer()