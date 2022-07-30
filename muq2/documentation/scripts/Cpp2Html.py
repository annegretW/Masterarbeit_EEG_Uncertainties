
import re
import markdown2

import sys

class TutorialDocument:

    def __init__(self, filename):

        # The regular expression that matches the special "/*** */" tutorial comment containing markdown
        self.exprString = '(\/\*\*\*.*?\*\/)'

        # Load the file
        with open(filename) as fin:
            self.fileContent = fin.read()

    def GetParts(self):
        """
        Splits the file contents into parts based on the tutorial comments.
        """
        pieces = re.split(self.exprString, self.fileContent, flags=re.S)
        return pieces

    def GetHeader(self):
        header =''
        return header

    def GetFooter(self):
        footer = ''
        return footer

    def FormatCode(self, string):
        if(string.isspace()):
            return '\n'
        else:
            code = self.StripLines(string.replace('<','&lt;').replace('>','&gt;'))
            output = '\n'
            if(len(code)>0):
                output = '\n<pre class="prettyprint lang-cpp">\n'
                output += code
                output += '\n</pre>\n'

            return output

    def FormatMarkdown(self, string):
        output = '\n'
        output += markdown2.markdown(string[4:-2].strip(), extras=["code-friendly"])
        output += '\n'

        return output

    def FormatCompleteCode(self, codePieces):
        stripPieces = []
        for piece in codePieces:
            if(len(piece)>0):
                stripPieces.append(piece.replace('<','&lt;').replace('>','&gt;').lstrip('\n').rstrip(' '))
        
        completeCode = self.StripLines(''.join(stripPieces))

        output = markdown2.markdown('#Complete Code')
        output += '\n<pre class="prettyprint lang-cpp">\n'
        output += completeCode
        output += '\n</pre>\n'

        return output


    def ToHTML(self):
        """
        Writes HTML containing a pretty version of the tutorial file
        """
        pieces = self.GetParts()

        output = self.GetHeader()

        codePieces = []

        # Loop through the pieces and format things for html
        for piece in pieces:
            isCode = True

            if(len(piece)>3):
                if(piece[0:4]=='/***'):
                    isCode = False

            if(isCode):
                output += self.FormatCode(piece)
                codePieces.append(piece)
            else:
                output += self.FormatMarkdown(piece)

        output += self.FormatCompleteCode(codePieces)

        output += self.GetFooter()

        return output

    def StripLines(self,s):
        lines = s.splitlines()
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        return '\n'.join(lines)

if __name__=='__main__':

    if(len(sys.argv)!=3):
        print("\nERROR: Incorrect number of command line arguments to Cpp2Html.py")
        print("\nUSAGE:\n\tpython Cpp2Html.py <c++ inputfile> <html outputfile>\n\n")

    assert(len(sys.argv)==3)
    filename = sys.argv[1]

    htmlString = TutorialDocument(filename).ToHTML()

    with open(sys.argv[2],'w') as fout:
        fout.write(htmlString)
