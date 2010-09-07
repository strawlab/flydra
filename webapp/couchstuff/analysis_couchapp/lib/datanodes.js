var parse_doc = function(doc) {
    
    var success = true;
    var props = [];
    var stat_tags = [];
    var value;

    switch (doc.type) {
    case "h5":
        // adapt to DataNode
        if (doc.has_2d_position) {
            props = props.concat( "2d position" );
        }
        if (doc.has_2d_orientation) {
            props = props.concat( "2d orientation" );
        }
        if (doc.has_3d_position) {
            props = props.concat( "3d position" );
        }
        if (doc.has_3d_orientation) {
            props = props.concat( "3d orientation" );
        }
        if (doc.has_calibration) {
            props = props.concat( "calibration" );
        }

	if (doc.filename === undefined) {
	    stat_tags = stat_tags.concat( 'unbuilt' );
	} else {
	    stat_tags = stat_tags.concat( 'built' );
	}

	sources = convert_h5_source_to_datanode_sources( doc.source );
        value = { properties : props,
                  start_time : doc.start_time,
                  stop_time  : doc.stop_time,
                  dataset    : doc.dataset,
		  sources    : sources,
		  status_tags: stat_tags,
		  comments   : doc.comments,
		  compute_start: doc.compute_start,
		  compute_stop: doc.compute_stop,
		  filename   : doc.filename,
		  filesize   : doc.filesize,
		  flydra_version : doc.flydra_version,
		  saved_images: doc.saved_images,
		  sha1sum    : doc.sha1sum
                }
        break;
    case "calibration":
        value = { properties : ["calibration"],
                  dataset    : doc.dataset,
                  sources    : [doc._id],
		  status_tags: ["built"],
                  start_time : doc.start_time,
                  stop_time  : doc.stop_time,
                }
        break;
    case "datanode":
        // already implements DataNode
        value = doc;
        break;
    default:
        success = false;
        break;
    }

    if (success) {
	return value;
    } else {
	return null;
    }
}

function convert_h5_source_to_datanode_sources( orig_source ) {

    // This method is an adapter to convert h5 doctype "source" field
    // to datanode doctype "sources" field. Doing this at all is
    // somewhat of a hack (h5 doc type sources should be just like
    // datanode sources). Furthermore, the string parsing
    // implementation below is horrible and will die if individual
    // source names have ", " in them, to name one example.

	computed_str = "computed from sources ";
	if (orig_source.slice(0,computed_str.length)==computed_str) {
	    sources_str = orig_source.slice(computed_str.length);

	    // Strip "[" and "]" from beginning and end of string
	    sources_str = sources_str.replace(RegExp('^\\[(.*)\\]$'),"$1");

	    // split at commas
	    source_strs = sources_str.split(", ");

	    // Strip "u'" and "'" from beginning and end of each straing
	    sources = [];
	    var re = RegExp("^u'(.*)'$");
	    for (var i in source_strs) {
		sources.push( source_strs[i].replace(re,"$1") );
	    }
	} else {
	    sources = [orig_source];
	}
	return sources;
}

