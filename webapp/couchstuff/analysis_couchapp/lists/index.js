function(head, req) {
    // !json templates.index
    // !code lib/JSON.js
    // !code vendor/couchapp/path.js
    // !code vendor/couchapp/template.js

    var indexPath = listPath('index','datasets');

    provides("html", function() {
        start({"headers":{"Content-Type" : "text/html; charset=utf-8"}});

        // render the html head using a template
        send(template(templates.index.head, {
            title : "Flydra analysis",
            index : indexPath,
            assets : assetPath()
        }));
        
        while (row = getRow()) {
            // XXX This only works for "datasets" view.
            //send( '<li>'+JSON.stringify(row) + '</li>\n' );
            var doc = row.value.dataset_doc;
            //var row_url = showPath('show_dataset', doc._id);

            send(template(templates.index.row, {
                link : showPath('show_dataset', doc._id),
                title : doc.name,
                dataset_size : "",
                summary : "",
                assets : assetPath()
            }));
            /*
            //var myshow_url = doc._id;
            var myshow_url = showPath('show_dataset', doc._id);
        
            send( '  <li> <a href="' + myshow_url + '">' + doc.name + '</a>\n' +
                  '       &#123; h5: {{h5_files}} files ({{h5_bytes_human}}),\n' +
                  '       ufmf: {{ufmf_files}} files ({{ufmf_bytes_human}}) &#125;\n' +
                  '  </li>\n');
            */
        };
        //send( '</ul>\n' );

        // render the html tail template
        return template(templates.index.tail, {
            assets : assetPath()
        });

    });
}
