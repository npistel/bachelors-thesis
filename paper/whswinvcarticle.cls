\ProvidesClass{whswinvcarticle}[2019/01/01 v 19.01 Article Class for W-HS WIN VC]
\LoadClass[paper=a4,fontsize=11pt]{article}

\PassOptionsToPackage{palatino=false,nochapters}{classicthesis}
\RequirePackage{whswinvc}

\patchcmd{\abstract}
  {\bfseries\abstractname}
  {\spacedlowsmallcaps{\abstractname}}
  {}{}

\let\origtitle\title
\def\whswinvc@title@short{}
\RenewDocumentCommand{\title}{o m}{%
  \origtitle{\LARGE\sffamily\textbf{#2}}%
  \IfValueTF{#1}{\def\whswinvc@title@short{#1}}{\def\whswinvc@title@short{#2}}%
}
\let\origauthor\author
\def\whswinvc@author@short{}
\RenewDocumentCommand{\author}{o m}{%
  \origauthor{\sffamily\textsc{#2}}%
  \IfValueTF{#1}{\def\whswinvc@author@short{#1}}{\def\whswinvc@author@short{#2}}%
}
\date{}

\AfterEndPreamble{
  \clearscrheadings
  \ohead[\whswinvc@course\\\whswinvc@semester@name]{\whswinvc@title@short}
  \ihead[\whswinvc@university\\\whswinvc@department]{\whswinvc@author@short}
  \cfoot[]{\pagemark}
  \ofoot[]{}
  \pagestyle{scrheadings}
  \thispagestyle{plain}
}

\endinput