function(head, req) {
    // !json templates.index
    // !code vendor/couchapp/path.js
    // !code lib/mustache.js
    // !code lib/human_sizes.js

    var indexPath = listPath('index','datasets');

    provides("html", function() {
        start({"headers":{"Content-Type" : "text/html; charset=utf-8"}});

        var row_view = {
            datasets : []
        }

        function size_format (filesize) {
            return getSize(filesize,1,0);
        }

        var make_dataset = function(r, path) {
            return {
                path : path,
                dataset_id : r.value.dataset_doc._id,
                dataset_name : r.value.dataset_doc.name,
                ufmf_bytes_human : size_format(r.value.ufmf_bytes),
                h5_bytes_human : size_format(r.value.h5_bytes),
                ufmf_files : r.value.ufmf_files,
                h5_files : r.value.h5_files
            };
        }

        while (row = getRow()) {
            //var doc = row.value.dataset_doc;
            var myshow_url = showPath('show_dataset', row.value.dataset_doc._id);
            row_view.datasets.push( make_dataset( row, myshow_url ));
        };

        var content = Mustache.to_html(templates.index.rows, row_view);

        var view = {
            title: "Flydra analysis",
            index: indexPath,
            content : content,
            asset_path : assetPath()
        }

        return Mustache.to_html(templates.index.must, view);

    });
}
