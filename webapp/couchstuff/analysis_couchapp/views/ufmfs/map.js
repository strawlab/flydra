function(doc) {
    if (doc.type=="dataset") {
        emit([doc._id,0], null);
    }
    if (doc.type=="ufmf") {
        if (doc.dataset != null) {
            emit([doc.dataset,1], null);
        }
    }
};
