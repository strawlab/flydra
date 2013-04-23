import sympy

S00=sympy.Symbol('S00')
S01=sympy.Symbol('S01')
S02=sympy.Symbol('S02')
S10=sympy.Symbol('S10')
S11=sympy.Symbol('S11')
S12=sympy.Symbol('S12')
S20=sympy.Symbol('S20')
S21=sympy.Symbol('S21')
S22=sympy.Symbol('S22')

x0=sympy.Symbol('x0')
x1=sympy.Symbol('x1')
x2=sympy.Symbol('x2')
y0=sympy.Symbol('y0')
y1=sympy.Symbol('y1')
y2=sympy.Symbol('y2')

post=sympy.Matrix(([y0-x0],[y1-x1],[y2-x2]))
S=sympy.Matrix(([S00,S01,S02],[S10,S11,S12],[S20,S21,S22]))
pre=post.T

maha3=(pre*S*post)[0,0]
(a,b,c,d,e,f,s)=[sympy.Symbol(x) for x in ('abcdefs')]
tmp = maha3
tmp = tmp.subs(y0,'a+s*d')
tmp = tmp.subs(y1,'b+s*e')
tmp = tmp.subs(y2,'c+s*f')
print 'solving with sympy',sympy.__version__
result = sympy.solve(sympy.diff(tmp,s),s)
print result
