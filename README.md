# xq - Simple CLI Excel/JSON/CSV Query Tool

XQ is a command-line tool for performing read-only queries on structured data files like `.xlsx`, `.json`, and `.csv`. It provides a simple command style interface to pandas dataframes - allowing you to quickly inspect data schemas and filter records directly from your terminal.

## Features

- **File Support**: Works with Excel (`.xlsx`), JSON (`.json`), and CSV (`.csv`) files.
- **Schema Inspection**: Quickly view column names, their data types, and generated short-name aliases.
- **Powerful Filtering**: A simple query language to filter records.
- **Wildcard Field Selection**: Use `*` and `?` patterns to match multiple fields at once.
- **Smart Caching**: Caches processed data and field name configurations for fast subsequent loads.
- **Flexible Output**: Automatic format detection based on terminal type, with manual override options.
- **Computed Fields**: Create derived columns with custom expressions using existing field data.

## Installation

1.  Clone the repository or download the `xq.py` script.
2.  Install the required Python dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

The basic invocation pattern is:

```bash
./xq.py --help
./xq.py [global_options] <path_to_file> <command> [command_options] [arguments...]
```

### Global+Command Options

| Option | Short | Description | Default |
|:-------|:------|:------------|:--------|
| `--format`  | `-f` | Output format (`table` or `json`) | Auto-detected based on TTY |
| `--verbose` | `-v` | Verbosity level (0-3) | 1 |

### Output Format Determination

XQ detects tty before outputting rich table format:

- **Interactive Terminal (TTY)**: Uses rich table format with colors and formatting
- **Non-TTY/Pipes**: Automatically switches to JSON format for programmatic use
- **Manual Override**: Use `--format` to explicitly set `table` or `json` format

Examples:
```bash
# Interactive use - shows rich table
./xq.py data.csv schema

# Piped output - automatically uses JSON
./xq.py data.csv flt "status=active,email" | cat

# Force JSON format even in terminal
./xq.py -f json data.csv schema
./xq.py data.csv schema -f json
```

### Home Folder Storage

XQ uses your home directory for storing configuration and cached data to improve performance.

-   **Configuration**: `~/.config/xq/<file_content_hash>.json`
    -   This file stores the generated short-name aliases for each unique data file you interact with. The file is named after the SHA1 hash of the file's content, so it remains persistent even if you move or rename the data file.
-   **Cache**: `~/.cache/xq/<file_content_hash>.pkl`
    -   This file stores the data from your file (e.g., the Excel sheet) as a processed Pandas DataFrame. This provides a significant speed-up for subsequent queries on the same file.

### Commands

#### `schema`

Displays the data schema, including column IDs (CID), full field names, generated short names, and data types.

**Usage:**

```bash
./xq.py path/to/your/data.xlsx schema
```

#### `flt`

Filters the data based on a query string and displays the results.

**Usage:**

```bash
./xq.py path/to/your/data.xlsx flt "query_string"
```

#### `tag ls`, `tag set`, `tag unset`

Manage computed fields (tags) which are new columns derived from existing data.

-   `tag set`: Creates a new or modifies an existing computed field.
-   `tag unset`: Removes a computed field.
-   `tag ls`: Lists all existing computed fields.

**Usage:**

```bash
# List all computed fields
./xq.py data.csv tag ls

# Create a new computed field
./xq.py data.csv tag set total-cost "{price} * {qty}"

# Remove a computed field
./xq.py data.csv tag unset total-cost
```

#### `cache clean`, `cache ls`

Manage cached data files for improved performance.

-   `<filename> cache clean`: Removes cached data and configuration for the specified file.
-   `cache clean`: Removes all cached data and configuration files.
-   `cache ls`: Lists all cache items with their SHA1 hashes, file sizes, and highlights the loaded file.

**Usage:**

```bash
# Clean cache for the loaded file
./xq.py data.csv cache clean

# List all cache items (no file required)
./xq.py cache ls

# Clean all cache items (no file required)
./xq.py cache clean
```

### Computed Fields (Tags)

When you create a computed field with `tag set`, you provide a name for the new field and an expression to calculate its value. This expression language is designed to be simple and intuitive, allowing you to reference other columns by their short names.

#### Tag Expression Language

**Syntax:**

The expression is a string that follows standard Python arithmetic and logical operations. To reference another field, enclose its **short name** in curly braces `{}`.

**Examples:**

Let's assume you have the following fields and short names:

| Field Name  | Short Name |
| :---------- | :--------- |
| `unit-price`| `price`    |
| `quantity`  | `qty`      |
| `tax-rate`  | `tax`      |

1.  **Calculate Total Cost**: Create a new field `total-cost` by multiplying `unit-price` and `quantity`.

    ```bash
    ./xq.py data.csv tag set total-cost "{price} * {qty}"
    ```

2.  **Calculate Price with Tax**: Create a `final-price` field that includes tax.

    ```bash
    ./xq.py data.csv tag set final-price "{price} * (1 + {tax})"
    ```

3.  **Conditional Logic (Future)**: *While not yet implemented, the language is designed to be extended with more complex logic, such as conditional statements.*

    ```bash
    # (Example of a possible future enhancement)
    # ./xq.py data.csv tag set category "if {price} > 100 then 'premium' else 'standard'"
    ```

This approach allows you to build powerful, readable expressions for data transformation directly from the command line.

## Filter Query Language

The filter query is a comma-separated string of expressions. Each expression can be:

1. **A filter condition** - applies a filter to the dataset
2. **A column name or alias** - includes the column in output
3. **A wildcard pattern** - matches multiple columns using `*` and `?`
4. **The special `*` wildcard** - includes all remaining columns

All filter conditions are combined with logical **AND**. Output columns appear in the order specified in the query string.

### Basic Syntax

```
field_expression[,field_expression,...]
```

Where each `field_expression` can be:
- `field_name` - include column in output
- `field_name operator value` - filter condition
- `pattern*` - wildcard pattern matching
- `*` - include all remaining columns

### Operators

| Operator | Description                                     | Example                                   |
| :------- | :---------------------------------------------- | :---------------------------------------- |
| `/.../`  | Shorthand for regex match                       | `'user-name/^A/'` or `'email/@gmail/'`   |
| `=`      | Equal to (numeric or string)                    | `'status=active'` or `'zip-code=90210'`   |
| `>`      | Greater than (numeric)                          | `'order-total>99.50'`                     |
| `<`      | Less than (numeric)                             | `'age<30'`                                |
| `>=`     | Greater than or equal to (numeric)              | `'score>=85'`                             |
| `<=`     | Less than or equal to (numeric)                 | `'inventory<=10'`                         |

### Wildcard Field Selection

XQ supports powerful wildcard patterns using standard shell globbing:

- `*` - matches any number of characters
- `?` - matches any single character
- `*` as standalone - includes all remaining columns

#### Wildcard Examples

**Field Names:**
```
user-id, user-name, user-email, order-id, order-total, order-date
```

**Wildcard Queries:**

1. **All user fields**: Select all columns starting with "user"
   ```bash
   ./xq.py data.csv flt "user-*"
   ```

2. **Mixed patterns**: Get user fields and any field ending with "total"
   ```bash
   ./xq.py data.csv flt "user-*,*-total"
   ```

3. **Filtered wildcards**: All user fields where order total > 100
   ```bash
   ./xq.py data.csv flt "user-*,order-total>100"
   ```

4. **Single character wildcards**: Fields like "user-a", "user-b", etc.
   ```bash
   ./xq.py data.csv flt "user-?"
   ```

5. **All remaining columns**: Show specific fields first, then everything else
   ```bash
   ./xq.py data.csv flt "user-id,user-name,*"
   ```

### Query Examples

#### Basic Filtering

-   **Simple Filter**: Show the `email` for all users whose `status` is 'active'.

    ```bash
    ./xq.py users.csv flt "status=active,email"
    ```

-   **Numeric Filter**: Show users with scores greater than 80.

    ```bash
    ./xq.py users.xlsx flt "score>80,user-name,score"
    ```

#### Filtering

-   **Shorthand Regex**: Show users whose first name starts with 'A'.

    ```bash
    ./xq.py users.csv flt "first-name/^A/,first-name,last-name"
    ```

-   **Email Domain Filter**: Find Gmail users.

    ```bash
    ./xq.py users.csv flt "email/@gmail/,user-name"
    ```

-   **Multiple Conditions**: First name starts with 'A' AND score > 80.

    ```bash
    ./xq.py users.xlsx flt "first-name/^A/,score>80,first-name,last-name,score"
    ```

-   **Combined Wildcard Selection**: All user fields for high-scoring users.

    ```bash
    ./xq.py users.csv flt "score>=90,user-*"
    ```

-   **Display All Columns**: Show all columns without filtering.

    ```bash
    ./xq.py inventory.json flt "*"
    ```

-   **Specific + Remaining**: Show key fields first, then everything else.

    ```bash
    ./xq.py inventory.json flt "product-id,price,*"
    ```

-   **Range Filtering**: Ages between 25 and 35.

    ```bash
    ./xq.py users.csv flt "age>=25,age<=35,user-name,age"
    ```

-   **Pattern Combinations**: User fields and order fields for active users.

    ```bash
    ./xq.py data.csv flt "status=active,user-*,order-*"
    ```

-   **Computed Field Filtering**: Using a computed field in queries.

    ```bash
    # First create the computed field
    ./xq.py data.csv tag set total-with-tax "{price} * (1 + {tax})"

    # Then filter using it
    ./xq.py data.csv flt "total-with-tax>50,product-name,total-with-tax"
    ```

### Error Handling

XQ provides clear error messages for common issues:

- **Invalid field names**: Clear indication of unrecognized fields
- **Type mismatches**: Helpful messages when applying numeric operators to string fields
- **Regex errors**: Specific feedback on malformed regular expressions
- **No matches**: Informative message when wildcard patterns don't match any fields

## Dependencies

- `pandas>=2.2.2` - Data manipulation and analysis
- `rich>=13.7.1` - Terminal formatting and tables
- `typer-di>=0.1.3` - CLI framework
- `openpyxl>=3.1.5` - Excel file support
