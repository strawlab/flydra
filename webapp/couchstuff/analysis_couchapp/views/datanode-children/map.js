function(doc) {
    // !code lib/datanodes.js
    var dndoc = parse_doc(doc);
    if (dndoc != null) {
	for (var i in dndoc.sources) {
	    emit(dndoc.sources[i], null);
	}
    }
};
