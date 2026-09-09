def sect(g, a, b, eps, maxit=500, trace=None):
    it = 0
    while abs(g(b)) >= eps and it < maxit:
        den = g(b) - g(a)
        if den == 0:
            if trace is not None:
                trace.append({'kind': 'sect_denzero',
                              'a': a, 'b': b,
                              'phi_a': g(a), 'phi_b': g(b)})
            break
        t = b - g(b) * (b - a) / den
        if trace is not None:
            trace.append({'kind': 'sect', 'a': a, 'b': b, 't': t,
                          'phi_a': g(a), 'phi_b': g(b)})
        if g(t) * g(a) > 0:
            a = t
        else:
            b = t
        it += 1
    return a if abs(g(a)) < abs(g(b)) else b
