function(doc) {
    if (doc.type=='job') {
	emit(doc._id, doc);
    }
};
