function(doc) {
    // !code lib/datanodes.js

    var value = parse_doc(doc);
    if (value!=null) {
	for (var i=0; i<value.properties.length; i++) {
	    emit( [value.dataset, value.properties[i]], value );
	}
    }
}
