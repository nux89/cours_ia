"""Sources des sept TP complémentaires. Génération explicite, sans écraser un TP existant."""
from pathlib import Path
import textwrap
import nbformat as nbf
import argparse

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--only', help='chemin relatif du seul TP à générer')
parser.add_argument('--overwrite', action='store_true', help='remplacer explicitement le TP sélectionné, sorties comprises')
args = parser.parse_args()


def notebook(path, title, introduction, sections, exercise, solution):
    if args.only and path != args.only:
        return
    cells = [nbf.v4.new_markdown_cell(f'# {title}\n\n{introduction}')]
    for heading, explanation, code in sections:
        cells.append(nbf.v4.new_markdown_cell(f'## {heading}\n\n{explanation}'))
        cells.append(nbf.v4.new_code_cell(textwrap.dedent(code).strip()))
    cells += [nbf.v4.new_markdown_cell('## Exercice\n\n' + exercise),
              nbf.v4.new_code_cell('# Écrivez votre expérience ici avant de lire la correction.'),
              nbf.v4.new_markdown_cell('## Correction et limites\n\n' + solution)]
    nb = nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python'}})
    nbf.validate(nb)
    dest = ROOT / path
    if dest.exists() and not args.overwrite:
        raise FileExistsError(f'TP existant conservé : {dest}')
    dest.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, dest)
    print(path)


notebook('09_raisonnement_symbolique_probabiliste/01_recherche_contraintes_bayes.ipynb',
         'A*, règles, contraintes et Bayes',
         'Objectif : produire un chemin optimal, une preuve par règles, une affectation faisable et une probabilité conditionnelle. Données synthétiques, bibliothèque standard uniquement. Lire le cours du module 9 ; aucun téléchargement.', [
('1. A* sur une grille', 'La file conserve g+h. Manhattan est adaptée aux quatre directions de coût unitaire. La table des coûts permet les améliorations.', '''
import heapq
from itertools import product
size = 6
walls = {(2, 0), (2, 1), (2, 2), (2, 3), (4, 3)}
start, goal = (0, 0), (5, 5)
def astar(heuristic):
    queue = [(heuristic(start), 0, start)]
    costs, parents = {start: 0}, {}
    expanded = 0
    while queue:
        _, cost, node = heapq.heappop(queue)
        if cost != costs[node]:
            continue
        expanded += 1
        if node == goal:
            path = [node]
            while node in parents:
                node = parents[node]
                path.append(node)
            return cost, path[::-1], expanded
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nxt = node[0] + dx, node[1] + dy
            if not (0 <= nxt[0] < size and 0 <= nxt[1] < size) or nxt in walls:
                continue
            proposal = cost + 1
            if proposal < costs.get(nxt, float('inf')):
                costs[nxt], parents[nxt] = proposal, node
                heapq.heappush(queue, (proposal + heuristic(nxt), proposal, nxt))
    return None, [], expanded
manhattan = lambda p: abs(p[0]-goal[0]) + abs(p[1]-goal[1])
cost, path, expanded = astar(manhattan)
reference = astar(lambda p: 0)
assert cost == reference[0] == len(path)-1
assert all(p not in walls for p in path)
assert all(abs(a[0]-b[0])+abs(a[1]-b[1]) == 1 for a,b in zip(path,path[1:]))
print({'coût A*': cost, 'états A*': expanded, 'états Dijkstra': reference[2], 'chemin': path})
'''),
('2. Chaînage avant et trace', 'Des faits nouveaux déclenchent les règles jusqu’au point fixe. La trace conserve les prémisses. L’absence de preuve ne signifie pas automatiquement fausseté.', '''
facts = {'temperature_haute', 'capteur_valide'}
rules = [({'temperature_haute', 'capteur_valide'}, 'alerte'), ({'alerte'}, 'inspection')]
trace = []
changed = True
while changed:
    changed = False
    for premises, conclusion in rules:
        if premises <= facts and conclusion not in facts:
            facts.add(conclusion)
            trace.append((sorted(premises), conclusion))
            changed = True
assert 'inspection' in facts and 'panne' not in facts
print(trace)
'''),
('3. CSP : détecter aussi l’impossibilité', 'Le backtracking teste uniquement les contraintes dont les variables sont affectées. Trois examens mutuellement incompatibles exigent trois créneaux.', '''
def solve_csp(domains, constraints, assignment=None):
    assignment = {} if assignment is None else assignment
    if len(assignment) == len(domains):
        return assignment.copy()
    variable = min((v for v in domains if v not in assignment), key=lambda v: len(domains[v]))
    for value in domains[variable]:
        candidate = {**assignment, variable: value}
        if all(a not in candidate or b not in candidate or candidate[a] != candidate[b]
               for a,b in constraints):
            result = solve_csp(domains, constraints, candidate)
            if result is not None:
                return result
    return None
constraints = [('A','B'), ('B','C'), ('A','C')]
assert solve_csp({v: [0,1] for v in 'ABC'}, constraints) is None
solution = solve_csp({v: [0,1,2] for v in 'ABC'}, constraints)
assert all(solution[a] != solution[b] for a,b in constraints)
print(solution)
'''),
('4. Réseau bayésien par énumération', 'Pluie et arrosage sont indépendants avant observation. On marginalise les configurations puis normalise. Le résultat attendu est calculé dans le cours.', '''
wet = {(1,1): .99, (1,0): .9, (0,1): .8, (0,0): .01}
joint = {}
for rain, sprinkler, moist in product([0,1], repeat=3):
    p_r = .2 if rain else .8
    p_s = .3 if sprinkler else .7
    p_m = wet[rain,sprinkler] if moist else 1-wet[rain,sprinkler]
    joint[rain,sprinkler,moist] = p_r*p_s*p_m
assert abs(sum(joint.values())-1) < 1e-12
p_wet = sum(p for (r,s,m),p in joint.items() if m)
p_rain_wet = sum(p for (r,s,m),p in joint.items() if r and m)
posterior = p_rain_wet/p_wet
assert abs(p_wet-.383) < 1e-12
assert abs(posterior - .1854/.383) < 1e-12
print({'P(route mouillée)': p_wet, 'P(pluie | route mouillée)': posterior})
''')],
'Remplacez Manhattan par zéro ; puis ajoutez une troisième couleur au CSP impossible. Pourquoi le graphe bayésien ne prouve-t-il pas un effet causal ?',
'h=0 donne Dijkstra et le même coût optimal. Trois couleurs rendent le triangle satisfaisable, ce que contrôle déjà la cellule 3. Le réseau factorise une loi observationnelle ; une interprétation causale exige des hypothèses supplémentaires. Ces petits problèmes ne mesurent pas le passage à l’échelle.')

notebook('10_methodes_specialisees/01_causalite_prevision.ipynb', 'Effet causal et prévision temporelle',
         'Deux expériences synthétiques indépendantes : confusion mesurée, puis prévision à un pas. Objectifs : identifier une hypothèse causale et vérifier la disponibilité temporelle des features. Dépendances : NumPy, Pandas, Matplotlib, scikit-learn.', [
('1. Simulation causale', 'X précède le traitement T et influence T et Y. L’effet injecté de T vaut 2. On compare association brute et régression ajustée, sous un modèle correctement spécifié.', '''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
rng = np.random.default_rng(42)
x = rng.normal(size=6000)
propensity = 1/(1+np.exp(-x))
treatment = rng.binomial(1, propensity)
outcome = 2*treatment + 3*x + rng.normal(size=len(x))
naive = outcome[treatment==1].mean()-outcome[treatment==0].mean()
adjusted = LinearRegression().fit(np.column_stack([treatment,x]), outcome).coef_[0]
assert abs(adjusted-2) < .2
assert abs(naive-2) > abs(adjusted-2)
print(pd.Series({'effet injecté': 2., 'différence brute': naive, 'ajustement sur X': adjusted}))
'''),
('2. Incertitude par bootstrap', 'Les unités sont indépendantes dans cette simulation. L’intervalle quantifie la variabilité de cet estimateur ; il ne teste pas l’absence de confusion cachée.', '''
boot = []
for _ in range(150):
    ids = rng.integers(0, len(x), len(x))
    boot.append(LinearRegression().fit(np.column_stack([treatment[ids],x[ids]]), outcome[ids]).coef_[0])
interval = np.quantile(boot, [.025,.975])
print('Intervalle bootstrap 95 % :', interval)
plt.hist(boot, bins=25)
plt.axvline(2, color='red', label='Effet injecté')
plt.xlabel('Effet ajusté'); plt.ylabel('Réplications'); plt.title('Simulation causale — variabilité bootstrap'); plt.legend(); plt.show()
'''),
('3. Série et caractéristiques disponibles', 'La cible est y(t). Les retards 1 et 7 sont connus au moment de prévoir t. Le calendrier futur est connu. Toutes les features sont construites avant le split mais sans lire le futur.', '''
dates = np.arange(500)
series = 10 + .015*dates + 2*np.sin(2*np.pi*dates/7) + rng.normal(0,.35,len(dates))
frame = pd.DataFrame({'target':series, 'time':dates})
frame['lag1'] = frame.target.shift(1)
frame['lag7'] = frame.target.shift(7)
frame['sin7'] = np.sin(2*np.pi*dates/7)
frame['cos7'] = np.cos(2*np.pi*dates/7)
frame = frame.dropna()
features = ['time','lag1','lag7','sin7','cos7']
assert frame.loc[20,'lag7'] == series[13]
'''),
('4. Validation à origines croissantes', 'Les blocs de validation se terminent avant la période test (t≥400). Le modèle est réajusté par bloc. On prévoit un pas à la fois : les observations intermédiaires sont donc disponibles.', '''
scores = []
for boundary in [250,300,350]:
    train = frame[frame.time < boundary]
    valid = frame[(frame.time >= boundary) & (frame.time < boundary+40)]
    model = make_pipeline(StandardScaler(), Ridge(alpha=1)).fit(train[features], train.target)
    for name,pred in [('Ridge',model.predict(valid[features])),('naïf',valid.lag1),('saisonnier',valid.lag7)]:
        scores.append({'origine':boundary,'modèle':name,'MAE':mean_absolute_error(valid.target,pred)})
print(pd.DataFrame(scores).pivot(index='origine',columns='modèle',values='MAE'))
'''),
('5. Test terminal', 'Le protocole et les hyperparamètres sont figés. MASE utilise uniquement le train pour sa normalisation. La figure montre le test, pas l’ajustement au train.', '''
train, test = frame[frame.time < 400], frame[frame.time >= 400]
model = make_pipeline(StandardScaler(), Ridge(alpha=1)).fit(train[features],train.target)
pred = model.predict(test[features])
scale = np.mean(np.abs(train.target-train.lag7))
assert train.time.max() < test.time.min() and scale > 0
print({'MAE Ridge':mean_absolute_error(test.target,pred), 'MASE Ridge':mean_absolute_error(test.target,pred)/scale,
       'MAE saisonnier':mean_absolute_error(test.target,test.lag7)})
plt.plot(test.time,test.target,label='Observé'); plt.plot(test.time,pred,label='Prévu à un pas')
plt.title('Série synthétique — test terminal'); plt.xlabel('Temps'); plt.ylabel('Valeur'); plt.legend(); plt.show()
''')],
'Retirez X du modèle causal. Pour la prévision, expliquez pourquoi fournir lag1 observé à tous les pas ne convient pas à une prévision de 100 pas émise une seule fois.',
'Sans X, la différence brute mélange sélection et effet causal. Pour 100 pas émis en t, y(t+1) est encore inconnu ; utiliser sa vraie valeur pour prévoir t+2 fuit. Il faut une stratégie récursive ou directe et un backtest correspondant à cet horizon.')

notebook('10_methodes_specialisees/02_recommandation_graphes.ipynb', 'Recommandation et GCN',
         'Deux jouets contrôlés : interactions implicites puis graphe assortatif. Les évaluations sont respectivement un masquage aléatoire et un protocole transductif. Elles ne remplacent pas une évaluation temporelle de production.', [
('1. Interactions et item masqué', 'Chaque utilisateur a un groupe de préférences. On masque un item pertinent avant de calculer les similarités, puis on exclut les items déjà vus des candidats.', '''
import numpy as np
import pandas as pd
import torch
from torch import nn
torch.set_num_threads(1)
rng = np.random.default_rng(42)
users, items = 120, 30
interactions = np.zeros((users,items))
held = []
for u in range(users):
    liked = rng.choice(np.arange((u%3)*10,(u%3+1)*10),6,replace=False)
    held.append(liked[0]); interactions[u,liked[1:]] = 1
held = np.array(held)
assert np.all(interactions[np.arange(users),held] == 0)
norm = np.linalg.norm(interactions,axis=0)
similarity = interactions.T@interactions/np.maximum(np.outer(norm,norm),1e-12)
np.fill_diagonal(similarity,0)
collaborative = interactions@similarity
popular = np.broadcast_to(interactions.sum(0),interactions.shape).copy()
def ranking_metrics(scores,k=5):
    scores = scores.copy(); scores[interactions > 0] = -np.inf
    ranked = np.argsort(-scores,axis=1,kind='stable')[:,:k]
    rel = ranked == held[:,None]
    assert not np.any(interactions[np.arange(users)[:,None],ranked])
    return {'Recall@5':rel.any(1).mean(), 'NDCG@5':(rel/np.log2(np.arange(k)+2)).sum(1).mean()}
print(pd.DataFrame({'popularité':ranking_metrics(popular), 'similarité items':ranking_metrics(collaborative)}).T)
'''),
('2. Graphe et split transductif', 'Deux communautés structurales et des features bruitées sont générées. Le train reçoit quelques labels seulement. Toutes les arêtes et features sont visibles : aucun label test n’est utilisé dans la loss.', '''
torch.manual_seed(42)
n = 120
labels = np.repeat([0,1],n//2)
prob = np.where(labels[:,None]==labels[None,:],.18,.015)
upper = np.triu(rng.random((n,n)) < prob,k=1)
adj = upper + upper.T
a = torch.tensor(adj,dtype=torch.float32)+torch.eye(n)
degree = a.sum(1)
normalized = degree.rsqrt()[:,None]*a*degree.rsqrt()[None,:]
x = torch.tensor(rng.normal(size=(n,6)),dtype=torch.float32)
x[:,0] += torch.tensor(labels,dtype=torch.float32)*.6
y = torch.tensor(labels,dtype=torch.long)
train_ids = torch.tensor(np.r_[rng.choice(60,12,False),rng.choice(np.arange(60,120),12,False)])
test_mask = torch.ones(n,dtype=torch.bool); test_mask[train_ids] = False
assert torch.isfinite(normalized).all()
'''),
('3. MLP versus GCN', 'La même taille de réseau et le même budget sont utilisés. Le MLP ignore la structure. La GCN applique deux propagations normalisées. Un graphe non homophile pourrait changer le résultat.', '''
def fit(use_graph):
    torch.manual_seed(7)
    first, second = nn.Linear(6,16), nn.Linear(16,2)
    optimizer = torch.optim.Adam(list(first.parameters())+list(second.parameters()),lr=.02,weight_decay=.01)
    def forward(features):
        hidden = torch.relu(first(normalized@features if use_graph else features))
        return second(normalized@hidden if use_graph else hidden)
    for _ in range(160):
        loss = nn.functional.cross_entropy(forward(x)[train_ids],y[train_ids])
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    with torch.no_grad():
        logits = forward(x)
        return (logits[test_mask].argmax(1)==y[test_mask]).float().mean().item()
print({'MLP test':fit(False),'GCN test transductif':fit(True)})
''')],
'Calculez NDCG@3 pour [1,0,1]. Expliquez pourquoi le protocole GCN n’est pas une généralisation à de nouveaux graphes.',
'NDCG=(1+1/log2(4))/(1+1/log2(3))≈0,920. Les nœuds test et leurs connexions sont visibles pendant l’apprentissage ; seuls leurs labels sont masqués. Pour l’inductif, réserver des graphes ou nœuds et leurs données selon le scénario de déploiement.')

notebook('03_deep_learning/06_transfert_fine_tuning.ipynb', 'Transfert et fine-tuning mesurables',
         'Préentraînement local sur digits 0–4, transfert vers 5–9. CPU, sans téléchargement. Voir ateliers_avances.md pour le protocole et les limites. Les budgets sont fixés avant le test ; aucun vainqueur n’est imposé.', [
('1. Données et modèle', 'Les classes source et cible sont disjointes. Le train cible est limité à 150 images ; le reste forme un test indépendant. Un MLP sert d’extracteur pour isoler le mécanisme de transfert.', '''
import copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
torch.set_num_threads(1); torch.manual_seed(42)
data = load_digits()
x = torch.tensor(data.data/16,dtype=torch.float32)
y = torch.tensor(data.target,dtype=torch.long)
source_x,source_y = x[y<5],y[y<5]
target_x,target_y = x[y>=5],y[y>=5]-5
train_ids,test_ids = train_test_split(np.arange(len(target_y)),train_size=150,stratify=target_y.numpy(),random_state=42)
assert set(train_ids).isdisjoint(test_ids)
def network():
    return nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Linear(32,5))
def train(model,inputs,targets,optimizer,steps):
    model.train()
    for _ in range(steps):
        loss = nn.functional.cross_entropy(model(inputs),targets)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    return loss.item()
source = network()
train(source,source_x,source_y,torch.optim.Adam(source.parameters(),lr=.01),180)
'''),
('2. Trois adaptations et contrôle des poids', 'La tête cible est réinitialisée identiquement pour les variantes. Le fine-tuning utilise un plus petit taux pour l’extracteur. On mesure les poids réellement modifiés.', '''
results = []
for mode in ['zéro','figé','fine-tuning']:
    torch.manual_seed(15)
    model = network()
    if mode != 'zéro':
        model[0].load_state_dict(copy.deepcopy(source[0].state_dict()))
    before = model[0].weight.detach().clone()
    if mode == 'figé':
        for p in model[0].parameters(): p.requires_grad_(False)
    optimizer = torch.optim.Adam([
        {'params':model[0].parameters(),'lr':.002 if mode=='fine-tuning' else .01},
        {'params':model[2].parameters(),'lr':.01}])
    final_loss = train(model,target_x[train_ids],target_y[train_ids],optimizer,180)
    changed = not torch.equal(before,model[0].weight)
    assert changed == (mode != 'figé')
    model.eval()
    with torch.no_grad():
        accuracy = (model(target_x[test_ids]).argmax(1)==target_y[test_ids]).float().mean().item()
    results.append({'stratégie':mode,'exactitude test':accuracy,'loss train':final_loss,'extracteur modifié':changed})
print(pd.DataFrame(results).set_index('stratégie'))
''')],
'Réduisez le nombre de labels cibles puis répétez avec plusieurs graines. Est-ce que figer signifie simplement ne pas appeler backward ?',
'Non : backward est nécessaire pour entraîner la tête ; requires_grad=False bloque les gradients des paramètres figés. Avec BatchNorm, le mode train peut modifier des statistiques même sans gradients ; ce MLP n’en contient pas. Mesurer une distribution de scores sans sélectionner la meilleure graine sur test.')

notebook('03_deep_learning/07_captioning_cnn_gru.ipynb', 'Captioning : CNN vers GRU',
         'Images synthétiques de formes colorées, légendes à deux attributs. Objectif : suivre image→état→tokens et comparer teacher forcing à génération autonome. CPU, aucune donnée externe.', [
('1. Images, tokens et cibles décalées', 'Les formes bougent et les pixels sont bruités. Le test utilise une autre graine. Toutes les combinaisons de concepts restent possibles dans les deux ensembles.', '''
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
torch.set_num_threads(1); torch.manual_seed(42)
vocab = ['PAD','BOS','EOS','carre','barre','rouge','vert']
def dataset(n,seed):
    rng = np.random.default_rng(seed)
    images = np.zeros((n,3,12,12),dtype=np.float32)
    captions = []
    for i in range(n):
        shape,color = rng.integers(0,2,size=2)
        row,col = rng.integers(1,5,size=2)
        height,width = (4,4) if shape==0 else (6,2)
        images[i,color,row:row+height,col:col+width] = 1
        images[i] += rng.normal(0,.025,images[i].shape)
        captions.append([3+shape,5+color,2])
    return torch.tensor(images),torch.tensor(captions)
x,y = dataset(400,42); xt,yt = dataset(100,43)
inputs = torch.cat([torch.ones(len(y),1,dtype=torch.long),y[:,:-1]],dim=1)
assert inputs.shape == y.shape == (400,3)
'''),
('2. Encodeur et décodeur', 'La représentation visuelle initialise la GRU. Les cibles sont forme, couleur, EOS. PAD est ignoré par la loss pour permettre une extension à longueurs variables.', '''
class Captioner(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3,8,3,padding=1),nn.ReLU(),nn.Flatten(),nn.Linear(8*12*12,32),nn.Tanh())
        self.embedding = nn.Embedding(len(vocab),16,padding_idx=0)
        self.gru = nn.GRU(16,32,batch_first=True)
        self.head = nn.Linear(32,len(vocab))
    def forward(self,image,tokens):
        hidden = self.encoder(image).unsqueeze(0)
        output,_ = self.gru(self.embedding(tokens),hidden)
        return self.head(output)
    def generate(self,image):
        hidden = self.encoder(image).unsqueeze(0)
        token = torch.ones(len(image),1,dtype=torch.long)
        predictions = []
        for _ in range(3):
            output,hidden = self.gru(self.embedding(token),hidden)
            token = self.head(output).argmax(-1)
            predictions.append(token)
        return torch.cat(predictions,dim=1)
model = Captioner()
opt = torch.optim.Adam(model.parameters(),lr=.008)
for step in range(240):
    ids = torch.randint(len(x),(64,))
    logits = model(x[ids],inputs[ids])
    loss = nn.functional.cross_entropy(logits.reshape(-1,len(vocab)),y[ids].reshape(-1),ignore_index=0)
    opt.zero_grad(); loss.backward(); opt.step()
assert logits.shape == (64,3,len(vocab))
'''),
('3. Générer sans vrais préfixes', 'Exact match exige les deux attributs et EOS. L’ablation sans image mesure l’apport du signal visuel. La génération a une borne de trois pas adaptée au vocabulaire du jouet.', '''
model.eval()
with torch.no_grad():
    predicted = model.generate(xt)
    blank = model.generate(torch.zeros_like(xt))
    exact = (predicted==yt).all(1).float().mean().item()
    blank_exact = (blank==yt).all(1).float().mean().item()
assert exact > .65, 'Le jouet doit apprendre les deux attributs ; examiner loss et images.'
print({'légendes exactes test':exact,'images supprimées':blank_exact})
fig,axes = plt.subplots(1,4,figsize=(10,3))
for i,ax in enumerate(axes):
    ax.imshow(xt[i].permute(1,2,0).clamp(0,1))
    ax.set_title(' '.join(vocab[j] for j in predicted[i].tolist() if j not in [0,1,2]))
    ax.axis('off')
fig.suptitle('Captioning synthétique — prédictions sur images test'); plt.tight_layout(); plt.show()
''')],
'Retenez une combinaison uniquement pour le test. Pourquoi un exact match élevé ici ne suffit-il pas pour des légendes de photographies ?',
'Le vocabulaire, les objets et la syntaxe sont minuscules ; le test est issu du même générateur. Un test de combinaison inédite examine une autre généralisation. Sur des photos, ajouter fidélité, paraphrases, comptage, relations spatiales et évaluations humaines.')

notebook('03_deep_learning/08_diffusion_2d.ipynb', 'DDPM : apprendre et échantillonner en 2D',
         'Apprentissage réel d’un débruiteur sur huit amas synthétiques, environ quelques secondes sur CPU. Aucun modèle d’image ni conditionnement textuel. Voir ateliers_avances.md pour les équations.', [
('1. Données et processus direct', 'Le produit des alpha devient proche de zéro au dernier temps. Les x_t sont construits directement depuis x_0 et un bruit connu.', '''
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
torch.set_num_threads(1); torch.manual_seed(42)
angles = torch.arange(8)*2*torch.pi/8
centers = torch.stack([torch.cos(angles),torch.sin(angles)],1)*1.5
def samples(n):
    return centers[torch.randint(8,(n,))] + .09*torch.randn(n,2)
data = samples(3000)
steps = 100
beta = torch.linspace(.0001,.12,steps)
alpha = 1-beta; abar = torch.cumprod(alpha,0)
assert abar[-1] < .01
def time_features(t):
    z = t.float()/steps
    return torch.stack([z,torch.sin(2*torch.pi*z),torch.cos(2*torch.pi*z)],1)
net = nn.Sequential(nn.Linear(5,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,2))
opt = torch.optim.Adam(net.parameters(),lr=.002)
history = []
for iteration in range(1200):
    x0 = data[torch.randint(len(data),(128,))]
    t = torch.randint(steps,(len(x0),))
    noise = torch.randn_like(x0)
    xt = abar[t,None].sqrt()*x0+(1-abar[t,None]).sqrt()*noise
    estimate = net(torch.cat([xt,time_features(t)],1))
    loss = nn.functional.mse_loss(estimate,noise)
    opt.zero_grad(); loss.backward(); opt.step()
    if iteration%100==0: history.append(loss.item())
print('Loss train aux étapes 0,100,… :',np.round(history,3))
'''),
('2. Test de débruitage et sampling inverse', 'Le test utilise des points frais. Le sampler part d’un bruit indépendant, applique la moyenne DDPM et sa variance postérieure, puis supprime le bruit au dernier pas.', '''
net.eval()
with torch.no_grad():
    x0 = samples(1000); t = torch.randint(steps,(1000,)); noise = torch.randn_like(x0)
    xt = abar[t,None].sqrt()*x0+(1-abar[t,None]).sqrt()*noise
    test_loss = nn.functional.mse_loss(net(torch.cat([xt,time_features(t)],1)),noise).item()
    baseline = noise.square().mean().item()
    generated = torch.randn(1200,2)
    for ti in reversed(range(steps)):
        t = torch.full((len(generated),),ti,dtype=torch.long)
        eps = net(torch.cat([generated,time_features(t)],1))
        mean = (generated-beta[ti]/(1-abar[ti]).sqrt()*eps)/alpha[ti].sqrt()
        previous = abar[ti-1] if ti > 0 else torch.tensor(1.)
        posterior_variance = beta[ti]*(1-previous)/(1-abar[ti])
        generated = mean + posterior_variance.sqrt()*torch.randn_like(mean) if ti > 0 else mean
assert test_loss < baseline and torch.isfinite(generated).all()
distance = torch.cdist(generated,centers)
print({'MSE bruit test':test_loss,'baseline zéro':baseline,'modes les plus proches présents':len(distance.argmin(1).unique()),
       'distance moyenne au centre le plus proche':distance.min(1).values.mean().item()})
'''),
('3. Comparaison visuelle', 'Conserver les mêmes axes pour comparer structure et dispersion. Couvrir tous les centres ne prouve pas que les densités sont correctement reproduites.', '''
fig,axes = plt.subplots(1,2,figsize=(9,4))
for ax,points,title in zip(axes,[data,generated],['Données synthétiques','DDPM — échantillons générés']):
    ax.scatter(points[:,0],points[:,1],s=3,alpha=.3)
    ax.set(xlim=(-2.5,2.5),ylim=(-2.5,2.5),xlabel='x1',ylabel='x2',title=title)
    ax.set_aspect('equal')
plt.tight_layout(); plt.show()
''')],
'Remplacez time_features par des zéros et réentraînez. Que mesure la loss de bruit, et que ne mesure-t-elle pas ?',
'Elle mesure une erreur de prédiction conditionnée par le processus de bruitage. Elle ne mesure pas directement diversité, fidélité ou mémorisation. Retirer le temps mélange des tâches de débruitage différentes. Comparer MSE, densité et modes avec les mêmes graines et un test fixe.')

notebook('04_ia_agentique/02_rag_documents_citations.ipynb', 'Récupérer des documents et citer les sources',
         'Baseline documentaire extractive sur les vrais Markdown du cursus. Pas de génération LLM ni API. Objectif : tester récupération, accès, provenance et abstention. Voir atelier_rag_documentaire.md.', [
('1. Ingestion locale et provenance', 'Les cours seuls sont indexés. Les fenêtres conservent les numéros de lignes, ce qui rend chaque extrait vérifiable. Aucun bloc de code du corpus n’est exécuté.', '''
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
root = Path.cwd().resolve()
while not (root/'README.md').exists() and root != root.parent:
    root = root.parent
files = sorted(root.glob('[0-9][0-9]_*/cours*.md'))
assert len(files) >= 9
chunks = []
for path in files:
    lines = path.read_text(encoding='utf-8').splitlines()
    for start in range(0,len(lines),24):
        text = '\\n'.join(lines[start:start+36])
        if text.strip():
            chunks.append({'file':str(path.relative_to(root)),'start':start+1,'end':min(start+36,len(lines)),'text':text})
vectorizer = TfidfVectorizer(strip_accents='unicode',ngram_range=(1,2),sublinear_tf=True)
matrix = vectorizer.fit_transform([c['text'] for c in chunks])
print({'documents':len(files),'passages':len(chunks)})
def retrieve(query,allowed=None,k=3):
    candidates = [i for i,c in enumerate(chunks) if allowed is None or c['file'] in allowed]
    if not candidates: return []
    scores = (matrix[candidates]@vectorizer.transform([query]).T).toarray().ravel()
    order = np.argsort(-scores,kind='stable')[:k]
    return [(float(scores[j]),chunks[candidates[j]]) for j in order]
'''),
('2. Calibration puis test', 'Le seuil est fixé sur deux questions répondables et un contrôle négatif de calibration. Le test comporte des questions différentes. La pertinence est annotée au niveau du document, pas du passage.', '''
calibration = [
    ('imputation données manquantes','01_nature_et_preparation_des_donnees/'),
    ('réseaux convolutifs CNN','03_deep_learning/'),
    ('zyxwvutsrqponmlk',None)]
positive_scores = [retrieve(q)[0][0] for q,target in calibration if target]
negative_scores = [retrieve(q)[0][0] for q,target in calibration if target is None]
threshold = (min(positive_scores)+max(negative_scores))/2
test_questions = [('corrélations Pearson Spearman','01_nature_et_preparation_des_donnees/'),
                  ('Q-learning Bellman','06_apprentissage_par_renforcement/'),
                  ('RAG bases de connaissances','04_ia_agentique/'),
                  ('abcdefghijkxyz',None)]
hits,abstentions = [],[]
for question,target in test_questions:
    found = retrieve(question)
    abstain = found[0][0] < threshold
    abstentions.append(abstain)
    if target: hits.append(any(c['file'].startswith(target) for _,c in found))
    print(question, '→', 'ABSTENTION' if abstain else found[0][1]['file'])
print({'seuil calibré':threshold,'Recall@3 documentaire':np.mean(hits)})
assert abstentions[-1]
'''),
('3. Réponse extractive et contrôle d’accès', 'La réponse cite les passages récupérés. La vérification compare chaque passage au fichier source. Le filtre est appliqué avant le classement.', '''
def answer(question,allowed=None):
    found = retrieve(question,allowed)
    if not found or found[0][0] < threshold:
        return 'Abstention : aucun passage suffisamment proche.'
    parts = []
    for score,c in found:
        source = (root/c['file']).read_text(encoding='utf-8').splitlines()
        assert c['text'] == '\\n'.join(source[c['start']-1:c['end']])
        citation = f"{c['file']}:L{c['start']}-L{c['end']}"
        parts.append(f"[{citation}] (similarité {score:.3f})\\n{c['text'][:450]}")
    return '\\n\\n'.join(parts)
print(answer('RAG bases de connaissances'))
allowed = {str(p.relative_to(root)) for p in files if p.parent.name.startswith('01_')}
restricted = retrieve('fine-tuning transfert',allowed)
assert all(c['file'] in allowed for _,c in restricted)
assert answer('abcdefghijkxyz').startswith('Abstention')
''')],
'Proposez une paraphrase sans les mots du titre ; ajoutez des questions hors domaine avec des mots ordinaires. Pourquoi le test négatif actuel est-il facile ?',
'La chaîne inconnue produit un vecteur nul ; elle ne teste pas les faux positifs lexicaux réalistes. Ajouter des contrôles négatifs proches, des annotations par passage et des questions contradictoires. Le TP mesure une récupération réelle et une réponse extractive, pas la résistance à l’injection ou la fidélité d’un LLM génératif.')
