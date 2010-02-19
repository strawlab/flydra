var JSON = JSON || {};

// implement JSON.stringify serialization
JSON.stringify = JSON.stringify || function (obj) {

	var t = typeof (obj);
	if (t != "object" || obj === null) {

		// simple data type
		if (t == "string") obj = '"'+obj+'"';
		return String(obj);

	}
	else {

		// recurse array or object
		var n, v, json = [], arr = (obj && obj.constructor == Array);

		for (n in obj) {
			v = obj[n]; t = typeof(v);

			if (t == "string") v = '"'+v+'"';
			else if (t == "object" && v !== null) v = JSON.stringify(v);

			json.push((arr ? "" : '"' + n + '":') + String(v));
		}

		return (arr ? "[" : "{") + String(json) + (arr ? "]" : "}");
	}
};


function(head, req) {
    provides("html", function() {
    start({"headers":{"Content-Type" : "text/html; charset=utf-8"}});
    send( '<ul>\n' );
    while (row = getRow()) {
        //send( JSON.stringify(row) +'\n');
        send( '  <li>' + row.value.name + '</li>\n' );
        };
    send( '</ul>\n' );
//    send( stringify( req ) + '\n\n');
   // return '<h1>hi</h1>'});
});
/*
    return{
        body : '' + doc.title + '',
        headers : {
            "Content-Type" : "application/xml",
        }
    }
*/
}
