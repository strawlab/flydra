# -*- coding: utf-8 -*-
"""
    sphinx.ext.graphviz
    ~~~~~~~~~~~~~~~~~~

    Allow graphviz-formatted graphs to be included in Sphinx-generated
    documentation inline.

    :copyright: 2008 Dell MessageOne, Inc.
    :license: BSD.
"""

import os
import sys
from subprocess import Popen, PIPE
try:
    from hashlib import sha
except ImportError:
    from sha import sha

from docutils import nodes
from docutils.parsers.rst import directives

_output_formats = {
#   'image/svg+xml': 'svg',
    'image/png': 'png',
    'application/postscript': 'ps',
}

class graphviz(nodes.General, nodes.Element):
    pass

def process_graphviz_nodes(app, doctree, docname):
    for node in doctree.traverse(graphviz):
        try:
            content = '\n'.join(node['graphviz_content'])
            filename = '%s' % sha(content).hexdigest()
            outfn = os.path.join(app.builder.outdir, '_images', 'graphviz', filename)
            if not os.path.exists(os.path.dirname(outfn)):
                os.makedirs(os.path.dirname(outfn))
            # iterate over the above-listed types
            for format_mime, format_ext in _output_formats.iteritems():
                graphviz_process = Popen([
                    getattr(app.builder.config, 'graphviz_dot', 'dot'),
                    '-T%s' % (format_ext,),
                    '-o', '%s.%s' % (outfn, format_ext), 
                ], stdin=PIPE)
                graphviz_process.stdin.write(content)
                graphviz_process.stdin.close()
                graphviz_process.wait()
                relfn = '_images/graphviz/%s' % (filename,)
            newnode = nodes.image()
            newnode['candidates'] = dict( [ (format_mime, '%s.%s' % (relfn, format_ext)) for (format_mime, format_ext) in _output_formats.iteritems() ] )
            # build PDF output from the previously generated postscript
            Popen([
                getattr(app.builder.config, 'graphviz_ps2pdf', 'ps2pdf'),
                '%s.ps' % (outfn,),
                '%s.pdf' % (outfn,)
            ]).wait()
            newnode['candidates']['application/pdf'] = '%s.pdf' % (outfn,)
            # and that's all, folks!
            node.replace_self(newnode)
        except Exception, err:
            from traceback import format_exception_only
            msg = ''.join(format_exception_only(err.__class__, err))
            newnode = doctree.reporter.error('Exception occured evaluating '
                                             'graphviz expression: \n%s' %
                                             msg, base_node=node)
            node.replace_self(newnode)

def graphviz_directive(name, arguments, options, content, lineno,
                    content_offset, block_txt, state, state_machine):
    node = graphviz()
    node['graphviz_content'] = content
    return [node]

def setup(app):
    app.add_node(graphviz)
    app.add_directive('graphviz', graphviz_directive, 1, (0, 0, 0))
    app.add_config_value('graphviz_dot', 'dot', False)
    app.add_config_value('graphviz_ps2pdf', 'ps2pdf', False)
    app.connect('doctree-resolved', process_graphviz_nodes)

# vim: sw=4 ts=4 sts=4 ai et
