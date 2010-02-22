function(resp) {
  var dbs = resp.rows.map(function(r) {
    return {
      db_name : r.key,
    };
  });
  return {dbs:dbs};
}
