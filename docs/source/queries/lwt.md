# Lightweight transaction (LWT) query

A lightweight transaction query can be expressed just like any other query, via `Session`, with the notable difference of having an additional consistency level parameter - the `serial_consistency_level`.


### Format of the query
A lightweight transaction query is not a separate type - it can be expressed just like any other queries: via `SimpleQuery`, `PreparedStatement`, batches, and so on. The difference lays in the query string itself - when it contains a condition (e.g. `IF NOT EXISTS`), it becomes a lightweight transaction. It's important to remember that CQL specification requires a separate, additional consistency level to be defined for LWT queries - `serial_consistency_level`. The serial consistency level can only be set to two values: `SerialConsistency::Serial` or `SerialConsistency::LocalSerial`. The "local" variant makes the transaction consistent only within the same datacenter. For convenience, Scylla Rust Driver sets the default consistency level to `LocalSerial`, as it's more commonly used. For cross-datacenter consistency, please remember to always override the default with `SerialConsistency::Serial`.
```rust
# extern crate scylla;
# use scylla::client::session::Session;
# use std::error::Error;
# async fn check_only_compiles(session: &Session) -> Result<(), Box<dyn Error>> {
use scylla::query::Query;
use scylla::statement::{Consistency, SerialConsistency};

// Create a Query manually to change the Consistency to ONE
let mut my_query: Query = Query::new("INSERT INTO ks.tab (a) VALUES(?) IF NOT EXISTS".to_string());
my_query.set_consistency(Consistency::One);
// Use cross-datacenter serial consistency
my_query.set_serial_consistency(Some(SerialConsistency::Serial));

// Insert a value into the table
let to_insert: i32 = 12345;
session.query_unpaged(my_query, (to_insert,)).await?;
# Ok(())
# }
```

The rest of the API remains identical for LWT and non-LWT queries.

See [Query API documentation](https://docs.rs/scylla/latest/scylla/statement/query/struct.Query.html) for more options

