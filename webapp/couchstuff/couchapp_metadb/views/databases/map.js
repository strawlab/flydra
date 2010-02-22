function(doc) {

    // This is structured like the map of tag-cloud in taskr, because
    // I could get the evently stuff to work then. The "doc._id" in
    // the key probably wouldn't be necessary if I could figure this
    // out.

    if (doc.type=="database") {
        emit([doc.database,1],doc);
    }
}
