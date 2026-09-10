def sect(g, a, b, eps, maxit=500, *, logger):
    it = 0
    while abs(g(b)) >= eps and it < maxit:
        phi_a = g(a)
        phi_b = g(b)
        den = phi_b - phi_a
        if den == 0:
            logger.log('sect_denzero', {'a': a, 'b': b,
                                        'phi_a': phi_a, 'phi_b': phi_b})
            break
        t = b - phi_b * (b - a) / den
        logger.log('sect', {'a': a, 'b': b, 't': t,
                            'phi_a': phi_a, 'phi_b': phi_b})
        if g(t) * phi_a > 0:
            a = t
        else:
            b = t
        it += 1
    return a if abs(g(a)) < abs(g(b)) else b