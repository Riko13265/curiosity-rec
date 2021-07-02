from math import exp

a = 0.2

x_l = [0, 0.5, 1]
sigma_x = 0.1

y_l = [0, 0.25, 0.5, 0.75, 1]
sigma_y = 0.1

rules = [(x1, x2, x3) for x1 in [0, 1, 2] for x2 in [0, 1, 2] for x3 in [0, 1, 2]]

rule_y_values = [0, 1, 1, 2, 3, 3, 4]

aa = a * a
ss = sigma_x * sigma_x

def rule_y(x1, x2, x3):
    return rule_y_values[x1 + x2 + x3]


# def gauss(x, avg, std):
#     n = (x - avg) / std
#     if abs(n) > 5:
#         return 0.
#     return exp(- pow(n, 2.))


def gauss(delta, std):
    n = delta / std
    if abs(n) > 5:
        return 0.
    return exp(- pow(n, 2.))


def xp_gauss(x_, x_l_):
    return \
        gauss((x_ - x_l_) * ss / (aa + ss), sigma_x) * \
        gauss((x_ - x_l_) * aa / (aa + ss), a)


# def omega(y, x, rule):
def omega(x, rule, xp_gausses):
    x_part = [xp_gausses[x[i], x_l[rule[i]]]
              for i in [0, 1, 2]]

    return x_part[0] * x_part[1] * x_part[2]


def fuzzy(x):
    xp_gausses = {(x_, x_l_): xp_gauss(x_, x_l_)
                   for x_ in x
                   for x_l_ in x_l}

    rule_omegas = [(rule, omega(x, rule, xp_gausses)) for rule in rules]
    sum_omega = sum(o for (r, o) in rule_omegas)

    if sum_omega == 0:
        return None

    return sum(rule_y(*r) * o
               for (r, o) in rule_omegas) / \
           sum_omega
