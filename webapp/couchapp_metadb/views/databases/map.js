function(doc) {
    if (doc.type=="database") {
        emit(doc.database,null);
    }
}
