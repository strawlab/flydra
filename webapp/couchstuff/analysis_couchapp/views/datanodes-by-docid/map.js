function(doc) {
    // !code lib/datanodes.js

    var value = parse_doc(doc);
    if (value!=null) {
	emit( doc._id, value );
    }
}
