function(doc) {
    if(doc.hasOwnProperty('type')) {
	if(doc.type=='junk') {
	    emit(doc._id, doc );
	}
	if(doc.junk) {
	    emit(doc._id, doc );
	}
    }
    else {
	emit(doc._id, doc );
    }
};
