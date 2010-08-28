function sources_selection_changed(event, ui) { 
    var result = $("#select-result").empty();
    var myid = this.id;
    $(".ui-selected", this).each(function(){
	    var index = $("#" + myid + " li").index(this);
	    result.append(" #" + (index + 1));
	});

 }


function no_space(str) {
    return str.replace(" ","_");
}

function FWA_create_ul_for_items( elem ) {
	var primary;

	if (typeof js_client_info_global.dominant_source_node_type === 'undefined') {
	    for (primary in js_client_info_global.sources) {
		// set the variable "primary" to the first key
		break;
	    }
	} else {
	    primary = js_client_info_global.dominant_source_node_type;
	}


	var myitems = "";
	for (var rownum in js_client_info_global.sources[primary].rows) {
	    var row = js_client_info_global.sources[primary].rows[rownum];
	    var myid = row.id;
	    myitems = myitems + "<li class=\"ui-widget-content\">" + myid + "</li>";
	}

	var my_id = "selectable_" + no_space(primary);
	elem.append("<h2>Select data "+primary+" sources</h2>");
	elem.append("You've selected: <span id=\"select-result\">none</span>.");
	elem.append("<p>Hold ctrl (on Mac: command) and click to select multiple sources.</p>");
	elem.append( "<div class=\"fwa_selectable\">")
	elem.append( "<ul id=\"" + my_id + "\" class=\"ui-selectable\"> " + myitems + "</ul>" );
	elem.append( "</div>" )
	$("#"+my_id).selectable({stop: sources_selection_changed});
}

$(document).ready(function(){

	if (typeof js_client_info_global === 'undefined') {
	    return;
	}

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
