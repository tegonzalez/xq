# xq - Simple CLI Excel/JSON/CSV Query Tool

XQ is a command-line tool for performing read-only queries on structured data files like `.xlsx`, `.json`, and `.csv`. It provides a simple command style interface to pandas dataframes - allowing you to quickly inspect data schemas and filter records directly from your terminal.

## Features

- **File Support**: Works with Excel (`.xlsx`), JSON, and CSV files.
- **Schema Inspection**: Quickly view column names, their data types, and generated short-name aliases.
- **Powerful Filtering**: A simple yet powerful query language to filter records.
- **Smart Caching**: Caches processed data and field name configurations for fast subsequent loads.
- **Flexible Output**: Display results as a clean, human-readable table (default) or as JSON for programmatic use.

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
./xq.py <path_to_file> <command> [arguments...]
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
python xq.py path/to/your/data.xlsx schema
```

#### `flt`

Filters the data based on a query string and displays the results.

**Usage:**

```bash
python xq.py path/to/your/data.xlsx flt "query_string"
```

### Filter Query Language

The filter query is a comma-separated string of expressions. Each expression can either be a **filter condition** or a **column name** to include in the output.

-   **Filter Conditions**: Apply a filter to the dataset.
-   **Column Names**: Specify a column to be included in the final output table.

All filter conditions are combined with a logical **AND**. The output columns will appear in the order they are listed in the query string.

#### Operators

| Operator | Description                                     | Example                                   |
| :------- | :---------------------------------------------- | :---------------------------------------- |
| `/.../`  | Shorthand for regex match                       | `'user-name/^A/'`                         |
| `=`      | Equal to (numeric or string)                    | `'status=active'` or `'zip-code=90210'`   |
| `>`      | Greater than (numeric)                          | `'order-total>99.50'`                     |
| `<`      | Less than (numeric)                             | `'age<30'`                                |
| `>=`     | Greater than or equal to (numeric)              | `'score>=85'`                             |
| `<=`     | Less than or equal to (numeric)                 | `'inventory<=10'`                         |

#### Examples

-   **Simple Filter**: Show the `email` for all users whose `status` is 'active'.

    ```bash
    python xq.py users.csv flt "status=active,email"
    ```

-   **Regex and Value Filter**: Show `first-name`, `last-name`, and `score` for all users whose first name starts with 'A' and whose score is greater than 80.

    ```bash
    python xq.py users.xlsx flt "first-name/^A/,last-name,score>80"
    ```

-   **Display Columns Only**: Show only the `product-id` and `price` columns for all records, without any filtering. The special `*` fieldname can be used to include all other fields.

    ```bash
    python xq.py inventory.json flt "product-id,price,*"
    ``` 