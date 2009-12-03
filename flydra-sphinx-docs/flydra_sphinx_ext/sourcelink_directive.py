from docutils.parsers.rst import directives
from sphinx.util.compat import Directive, directive_dwim
from sphinx import addnodes

class Sourcelink(Directive):
    """
    Directive to create a centered line of bold text.
    """

    has_content = False
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {}

    def run(self):
        if not self.arguments:
            return []
        subnode = addnodes.centered()
        inodes, messages = self.state.inline_text(self.arguments[0],
                                                  self.lineno)
        subnode.extend(inodes)
        return [subnode] + messages

#directives.register_directive('sourcelink', directive_dwim(Sourcelink))

def setup(app):
    app.add_directive('sourcelink',Sourcelink)
