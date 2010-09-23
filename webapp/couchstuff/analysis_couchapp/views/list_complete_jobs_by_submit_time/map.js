function(doc) {
    if (doc.type=='job') {
	if (doc.state=='complete') {
	    emit(doc.submit_time, doc);
	}
    }
};
