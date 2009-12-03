from docutils.parsers.rst import directives
from sphinx.util.compat import Directive, directive_dwim
from sphinx import addnodes
from docutils.parsers.rst import states

class Sourcelink(Directive):
    """
    Directive to create a centered line of bold text.
    """

    has_content = False
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = True
    option_spec = {'command':directives.unchanged,
                   }

    def run(self):
        if 0:
            # force this to be a sustitution ref
            if not isinstance(self.state, states.SubstitutionDef):
                error = self.state_machine.reporter.error(
                    'Invalid context: the "%s" directive can only be used '
                    'within a substitution definition.' % (self.name),
                    nodes.literal_block(block_text, block_text), line=lineno)
                return [error]

        if not len(self.arguments)==1:
            raise self.error(
                'Error in "%s" directive: need exactly one argument'
                % (self.name,))

        tdict = self.options.copy()
        tdict['file']=self.arguments[0]
        if tdict.has_key('alt'):
            del tdict['alt']

        # like sorted iteritems()
        tstrs = []
        keys = tdict.keys()
        keys.sort()
        for k in keys:
            v = tdict[k]
            tstrs.append( '%s:%s' % (k,v) )

        tstr = ', '.join(tstrs)

        inodes, messages = self.state.inline_text( 'source(%s)'%tstr,
                                                   self.lineno )
        return inodes + messages

def setup(app):
    app.add_directive('sourcelink',Sourcelink)
