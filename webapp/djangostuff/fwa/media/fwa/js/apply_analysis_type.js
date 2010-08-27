function sources_selection_changed(event, ui) { 
    console.log("selected 3");
 }

function FWA_create_ul_for_items( elem ) {
	var primary;

	if (typeof js_client_info_global.dominant_source_node_type === 'undefined') {
	    // TODO: choose the one with most options
	    primary = js_client_info_global.sources[0];
	} else {
	    primary = js_client_info_global.dominant_source_node_type;
	}


	var myitems = "";
	for (var rownum in js_client_info_global.sources[primary].rows) {
	    var myid = js_client_info_global.sources[primary].rows[rownum]._id
	    myitems = myitems + "<li class=\"ui-widget-content\">" + myid + "</li>";
	}
	elem.append("<h2>Select data sources</h2>");
	elem.append("<p>Hold ctrl (on Mac: command) and click to select multiple sources.</p>");
	elem.append( "<ul id=\"selectable\"> " + myitems + "</ul>" );
	$("#selectable").selectable({selected: sources_selection_changed,
		    unselected: sources_selection_changed});
}

$(document).ready(function(){

	// With this javascript running, disable user input into all but
	// dominant selection box. This code will fill the rest.

	for (var source_node_type in js_client_info_global.sources) {

	    var eid = js_client_info_global.sources[source_node_type].select_id;
	    var selector = "#" + eid;
	    var dom = $(selector);
	    // dom is just the <select> box, hide its parent (the <p>)
	    dom.parent().hide();
	}
	
	FWA_create_ul_for_items($("#fwa_jq_div"));
});
