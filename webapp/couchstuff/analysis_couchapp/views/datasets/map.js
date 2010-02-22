function(doc) {
    if (doc.type=="dataset") {
        emit(doc._id, doc.name);
    }
};
