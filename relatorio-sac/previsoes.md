# Previsões — Partes 4 e 5

Escritas antes de rodar os experimentos correspondentes, como o enunciado pede.
Base: diagnóstico do config inicial (alpha estabiliza em ~0,09; entropia segue o alvo
de -3,0; q_minus_return oscila em torno de zero; Q1 ainda subindo aos 100k).

## Parte 4, ablação 1 — `autotune=false` com alpha fixo

Escolho **alpha = 0,09**, que é o patamar em que o autotune estabilizou nas três seeds
do baseline depois dos 20 mil passos.

Previsão: como o valor fixo é o mesmo que o autotune encontraria, espero diferença
pequena no retorno. A diferença deve aparecer no **começo**: o autotune parte de 0,90 e
leva uns 20 mil passos para descer, então ele explora muito mais nessa fase inicial,
enquanto o alpha fixo em 0,09 já começa "frio". Espero então que a ablação suba mais
rápido no início e termine parecida ou um pouco pior, porque perdeu exploração cedo.
Espero também que `losses/entropy` deixe de colar no `target_entropy`, já que não há
mais nada forçando isso: sem o controlador, a entropia vira consequência e não alvo.

## Parte 4, ablação 2 — `clipped_double_q=false`

Previsão: é a ablação que deve doer mais. Sem o mínimo entre os dois críticos, o alvo
passa a usar só Q1, e o erro de estimativa deixa de ser cortado pelo lado pessimista.
Espero `losses/qf1_values` crescendo bem acima do baseline e, principalmente,
`charts/q_minus_return` ficando **positivo e crescente**, que é a assinatura da
superestimação. No baseline ele oscila em torno de zero. O retorno deve cair, porque o
ator maximiza um Q inflado e acaba preferindo ações que o crítico superestima.

## Parte 5 — SAC contra PPO contínuo, 100k passos

Previsão: o SAC deve ganhar com folga em eficiência amostral. Ele é off-policy com
replay, então cada transição entra em muitos passos de gradiente, enquanto o PPO
descarta o rollout depois das poucas épocas. Com apenas 100 mil passos, esse é
exatamente o regime onde o replay compensa.

Em tempo de relógio espero o contrário, pelo mesmo motivo que vimos no PPO contra A2C:
o SAC faz um passo de gradiente por passo de ambiente (dois, se o candidato de
`updates_per_step=2` vencer), então ele é muito mais lento por passo coletado. Se o
config final tiver updates_per_step=2, a desvantagem de tempo dobra.

Sobre reuso: no SAC, com buffer de 1M e 100k passos, nada é descartado, e com
batch 256 e um update por passo cada transição entra em cerca de 256 passos de
gradiente ao longo do treino. No PPO contínuo, cada transição entra em
update_epochs vezes e depois é jogada fora. A diferença deve ser de duas ordens de
grandeza.
