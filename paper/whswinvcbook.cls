\ProvidesClass{whswinvcbook}[2019/01/01 v 19.01 Book Class for W-HS WIN VC]
\PassOptionsToPackage{headinclude,footinclude,paper=a4,BCOR=10mm}{typearea}
\LoadClass[oneside,openright,titlepage,numbers=noenddot,cleardoublepage=empty,BCOR=10mm,paper=a4,fontsize=11pt]{scrbook}

\PassOptionsToPackage{style=classicthesis, palatino=false, floatperchapter}{classicthesis}
\RequirePackage{whswinvc}
\setlength{\cftbeforechapskip}{\baselineskip}

\RequirePackage[perpage, symbol]{footmisc}
\deffootnote[1em]{1em}{1em}{\textsuperscript{\thefootnotemark}\ }
\RequirePackage{scrlayer-scrpage}
\chead[]{\headmark}
\cfoot[\pagemark]{\pagemark}

%%% define spacedsmallcaps
%%% engine-dependent letterspacing: microtype for pdflatex, fontspec for xelatex and luatex
\ifthenelse{\boolean{pdf}\OR\boolean{luatex}}{ % pdf will soon test true for both pdflatex and luatex; \AND\NOT\boolean{luatex}
  \PackageInfo{whswinvcbook}{Using microtype for character spacing.%
    Make sure your pdflatex is version 1.40 or higher.}%
  % \microtypesetup{expansion=false}
  \microtypesetup{protrusion=true,expansion=true}
  \DeclareRobustCommand{\spacedsmallcaps}[1]{\textls[80]{\ct@caps{#1}}}
}{\relax}

\ifthenelse{\boolean{xetex}}{ % use fontspec features; \OR\boolean{luatex} does not work well
  \PackageInfo{whswinvcbook}{Using fontspec for character spacing.}%
  \microtypesetup{expansion=false}
  \DeclareRobustCommand{\spacedsmallcaps}[1]{{\addfontfeatures{LetterSpace=14.0}\ct@caps{#1}}}% WordSpace=1.8
}{\relax}

\addtokomafont{subject}{\normalfont}
\addtokomafont{author}{\scshape}

\extratitle{\centering%
  \spacedlowsmallcaps{\whswinvc@course@abbr{} -- \whswinvc@semester@abbr}\\[0.618\baselineskip]
  \spacedlowsmallcaps{\whswinvc@author@short}\\[0.618\baselineskip]
  \expandafter\spacedsmallcaps{\whswinvc@title@short}
}
\subject{\spacedlowsmallcaps{\whswinvc@course}}
\date{\whswinvc@semester@name}
\publishers{%
  \scshape\MakeLowercase{\whswinvc@university}\\%
  \scshape\MakeLowercase{\whswinvc@department}%
}


\titlehead{}

\let\whswinvc@orig@author\author
\def\whswinvc@author@short{}
\RenewDocumentCommand{\author}{o m}{%
  \whswinvc@orig@author{#2}%
  \IfValueTF{#1}{\def\whswinvc@author@short{#1}}{\def\whswinvc@author@short{#2}}%
}

\let\whswinvc@orig@title\title
\def\whswinvc@title@short{}
\RenewDocumentCommand{\title}{o m}{%
  \whswinvc@orig@title{\LARGE\sffamily\textbf{#2}}%
  \IfValueTF{#1}{\def\whswinvc@title@short{#1}}{\def\whswinvc@title@short{#2}}%
}

\let\whswinvc@orig@subtitle\subtitle
\RenewDocumentCommand{\subtitle}{m}{%
  \whswinvc@orig@subtitle{\spacedlowsmallcaps{#1}}%
}



\NewDocumentCommand{\tableofs}{}{%
  \cleardoublepage
  \manualmark
  \markboth{\spacedlowsmallcaps{\contentsname}}{\spacedlowsmallcaps{\contentsname}}%
  \tableofcontents

  \cleardoublepage
  \manualmark
  \markboth{\spacedlowsmallcaps{\listfigurename}}{\spacedlowsmallcaps{\listfigurename}}
  \listoffigures

  \cleardoublepage
  \manualmark
  \markboth{\spacedlowsmallcaps{\listfigurename}}{\spacedlowsmallcaps{\listtablename}}
  \listoftables
}

\let\whswinvc@orig@mainmatter\mainmatter
\RenewDocumentCommand{\mainmatter}{}{%
  \whswinvc@orig@mainmatter
  \automark[section]{chapter}
  \renewcommand{\chaptermark}[1]{\markboth{\spacedlowsmallcaps{##1}}{\spacedlowsmallcaps{##1}}}
  \renewcommand{\chaptermark}[1]{\markboth{\spacedlowsmallcaps{##1}}{\spacedlowsmallcaps{##1}}}
  \begingroup
  \titlespacing*{\chapter}{0pt}{0.381\textwidth}{1.618\baselineskip}
}
\let\whswinvc@orig@backmatter\backmatter
\RenewDocumentCommand{\backmatter}{}{%
  \endgroup
  \whswinvc@orig@backmatter
}
\NewDocumentCommand{\preparebibliography}{}{%
  \manualmark
  \markboth{\spacedlowsmallcaps{\bibname}}{\spacedlowsmallcaps{\bibname}}
}


\def\whswinvc@chapter@principal@author@text@de{Hauptautor:~}
\def\whswinvc@chapter@principal@author@text@en{principal author:~}
\let\whswinvc@orig@chapter\chapter
\RenewDocumentCommand{\chapter}{s o m g}{%
  \IfLanguagePatterns{ngerman}{%
    \def\whswinvc@chapter@principal@author{\whswinvc@chapter@principal@author@text@de}%
  }{%
    \IfLanguagePatterns{english}{%
      \def\whswinvc@chapter@principal@author{\whswinvc@chapter@principal@author@text@en}%
    }{}%
  }%
  \IfBooleanTF{#1}{%
    \IfValueTF{#2}{%
      \IfValueTF{#4}{%
        \whswinvc@orig@chapter*[#2]{#3\,\mbox{\protect\footnotemark}}%
        \footnotetext{\whswinvc@chapter@principal@author#4}%
      }{%
        \whswinvc@orig@chapter*[#2]{#3}%
      }%
    }{%
      \IfValueTF{#4}{%
        \whswinvc@orig@chapter*[#3]{#3\,\mbox{\protect\footnotemark}}%
        \footnotetext{\whswinvc@chapter@principal@author#4}%
      }{%
        \whswinvc@orig@chapter*{#3}%
      }%
    }%
  }{
    \IfValueTF{#2}{%
      \IfValueTF{#4}{%
        \whswinvc@orig@chapter[#2]{#3\,\mbox{\protect\footnotemark}}%
        \footnotetext{\whswinvc@chapter@principal@author#4}%
      }{%
        \whswinvc@orig@chapter[#2]{#3}%
      }%
    }{%
      \IfValueTF{#4}{%
        \whswinvc@orig@chapter[#3]{#3\,\mbox{\protect\footnotemark}}%
        \footnotetext{\whswinvc@chapter@principal@author#4}%
      }{%
        \whswinvc@orig@chapter{#3}%
      }%
    }%
  }%
}

\endinput
