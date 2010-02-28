function(doc) {
    // !code lib/datanodes.js

    var value = parse_doc(doc);
    if (value!=null) {
        emit( [value.dataset, value.properties, value.status_tags], value );
    }
}
