# Previsões (escritas antes de rodar qualquer varredura)

Wagner Victor Alves de Menezes e Victor Ribeiro Jacome

Baseline: CartPole-v1, 500k passos, num_envs=8, num_steps=5, gamma=0.99, lr=7e-4,
ent_coef=0.01, vf_coef=0.5, use_baseline=true.

## Q1 — num_envs (1, 8, 32)

O batch de cada atualização é num_envs x num_steps, então com 1 ambiente ele cai para 5
transições. Espero um `policy_loss` visivelmente mais ruidoso nesse caso, porque a média
de 5 amostras estima muito mal o gradiente esperado, e uma curva de retorno mais lenta e
instável. Com 32 ambientes o gradiente fica mais limpo, mas em 500k passos cabem 4 vezes
menos atualizações que no baseline, então acho que ele não vai ser 4 vezes melhor e pode
até terminar atrás. O `SPS` deve subir bastante com mais ambientes, porque dar um passo
em 32 ambientes vetorizados custa quase o mesmo que em 8.

## Q2 — num_steps (1, 5, 32, 128)

Com num_steps=1 o alvo é quase todo bootstrap, r + gamma*V(s'), então tem pouca variância
e muito viés vindo de um crítico que ainda está errado. Com 128 é quase Monte Carlo, com
pouco viés e muita variância. Espero o `value_loss` pequeno no 1 e grande no 128, já que a
escala do alvo cresce junto com o horizonte. Minha aposta é que o baseline de 5 fica no
melhor ponto e que o 128 aprende de forma mais errática. Desconfio da armadilha que o
enunciado avisa: no num_steps=1 a `explained_variance` deve parecer ótima, mas o alvo ali
é quase o próprio V(s), então o crítico está sendo medido contra si mesmo.

## Q3 — ent_coef (0, 0.01, 0.1)

Com ent_coef=0,1 o bônus domina a perda e a política é empurrada para ficar uniforme.
Espero a entropia parada perto de ln 2, que é 0,693, e o retorno estacionando bem abaixo
de 500, porque o agente é premiado por continuar aleatório. Com ent_coef=0 a entropia deve
despencar rápido e a política virar determinística cedo. No CartPole isso pode não ser
fatal, já que são só duas ações e a recompensa é densa, mas espero mais diferença entre as
duas seeds do que nas outras configurações, com pelo menos uma correndo risco de travar.

## Q4 — use_baseline=false

Sem o baseline o peso passa a ser o retorno bruto. No CartPole toda recompensa é +1, então
esse peso é sempre positivo e grande, na casa das dezenas, enquanto a vantagem fica
centrada em zero. Espero `advantage_mean` saltando de perto de zero para bem positivo e
`advantage_std` várias vezes maior, que é o resultado teórico da aula: o baseline não muda
o gradiente esperado, só a variância. Como todo peso é positivo, toda ação amostrada tem a
probabilidade empurrada para cima, e a que foi amostrada mais vezes ganha. Isso deve fazer
a entropia colapsar mais rápido que no baseline e a curva de retorno ficar mais instável.
