# Which extensions to build alongside DuckDB. Consumed via -DDUCKDB_EXTENSION_CONFIGS.
duckdb_extension_load(rocket
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}
    LOAD_TESTS
)
