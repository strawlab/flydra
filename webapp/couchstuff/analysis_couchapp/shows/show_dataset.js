function(doc,req){
    // !json templates
    // !code vendor/couchapp/path.js
    // !code lib/mustache.js
    // !code lib/JSON.js

    var indexPath = showPath('show_datasets');

    provides("html", function() {
        start({"headers":{"Content-Type" : "text/html; charset=utf-8"}});

        content = JSON.stringify({req:req,
                                  doc:doc})

        var view = {
            title: "Flydra analysis",
            index: indexPath,
            content : content,
            asset_path : assetPath()
        }

        return Mustache.to_html(templates.base, view);

    });
}