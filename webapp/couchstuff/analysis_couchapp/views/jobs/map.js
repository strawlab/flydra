function(doc) {
    if (doc.type=='job') {
	emit(doc.state, doc.id);
    }
};
