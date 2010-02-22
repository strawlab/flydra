function(doc) {
    if (doc.type=="dataset") {
        emit([doc._id,0], null);
    }
    if (doc.type=="h5") {
        if (doc.source=="original data") {
            emit([doc.dataset,1], null);
        }
    }
};
