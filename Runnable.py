from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough


sign = lambda x: '+' if int(x)>=0 else '-'


def solve_quadratic (inputs):
    if isinstance(inputs['sqr_D'], complex):
        return "Действительных корней нет"
    elif inputs['sqr_D'] == 0: 
        return RunnableLambda(lambda inputs: f"Единственный корень: {(-inputs['b']) / (2 * inputs['a'])}")
    else:
        lambda_1 = RunnableLambda(lambda inputs: (-inputs['b'] + inputs['sqr_D']) / (2 * inputs['a']))
        lambda_2 = RunnableLambda(lambda inputs: (-inputs['b'] - inputs['sqr_D']) / (2 * inputs['a']))
        lambds = RunnableParallel(lambda_1=lambda_1, lambda_2=lambda_2)
        output = RunnableLambda(lambda inputs: f"Корни: {inputs['lambda_1']} и {inputs['lambda_2']}")
        return lambds | output



D = RunnableLambda(lambda inputs: inputs['b']**2 - 4 * inputs['a'] * inputs['c'])
square = RunnableLambda(lambda D: D**(1/2))
main = RunnableLambda(solve_quadratic)
output_size = RunnableLambda(lambda input: f"Для квадратного уравнения {sign(input['a'])}{abs(input['a'])}x²{sign(input['b'])}{abs(input['b'])}x{sign(input['c'])}{abs(input['c'])} Ответом является:\n{input['lamds']}")

discriminant = RunnablePassthrough.assign(lamds = RunnablePassthrough.assign(sqr_D = D | square) | main) | output_size


tests = [ {'a':1, 'b':-4, 'c': 3}, {'a':1, 'b':-4, 'c': 4}, {'a':1, 'b':-4, 'c': 5}]

for test in tests:
    print(discriminant.invoke(test), end = '\n\n')

"""
Для квадратного уравнения +1x²-4x+3 Ответом является:
Корни: 3.0 и 1.0

Для квадратного уравнения +1x²-4x+4 Ответом является:
Единственный корень: 2.0

Для квадратного уравнения +1x²-4x+5 Ответом является:
Действительных корней нет
"""
