function(resp) {
  var datasets = resp.rows.map(function(r) {
    return {
      dataset_id : r.key,
      dataset_name : r.value,
    };
  });
  return {datasets:datasets};
}
