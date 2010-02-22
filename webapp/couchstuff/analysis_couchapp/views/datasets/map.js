function(doc) {
    if (doc.type=="dataset") {
        emit([doc._id, 0], doc);
    } else if (doc.type=="ufmf") {
        emit([doc.dataset, 1], doc.filesize);
    } else if ((doc.type=="h5") && (doc.source=="original data")) {
        emit([doc.dataset, 2], doc.filesize);
    }
};
