function(doc) {
    if (doc.type=='job') {
	emit(doc.datanode_id, null);
    }
};
