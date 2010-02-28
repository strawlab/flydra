function(doc) {
    // !code lib/datanodes.js

    var value = parse_doc(doc);
    if (value!=null) {
	for (var i=0; i<value.properties.length; i++) {
	    emit( doc._id, value );
	}
    }
}
