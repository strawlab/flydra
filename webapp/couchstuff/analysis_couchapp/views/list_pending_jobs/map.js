function(doc) {
    if (doc.type=='job') {
	if (doc.state!='complete') {
	    emit(doc.state, doc);
	}
    }
};
