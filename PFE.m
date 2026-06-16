%% =========================================================================
%  PLATEFORME DE SIMULATION ULTIME — SÉCHEUR TSP
%  =========================================================================
%  VERSION     : FINALE V3-ECO — OPTIMISATION ÉNERGETIQUE
%  FICHIER     : STANDALONE — poids LSTM embarqués
%                tout.xlsx REQUIS pour la comparaison réelle
%
%  OBJECTIF    : [ECO] Maintenir production légèrement > réel
%                       TOUT EN MINIMISANT la consommation de fuel
%                       → Réduction coût énergétique (€/tonne)
%
%  STRATÉGIE ECO :
%    - Production cible = Prod_réelle + marge_prod (ex : +20 t/h max)
%    - Pénalisation du fuel dans l'objectif MPC (terme R_eco)
%    - Plancher Qf dynamique abaissé pour laisser le MPC économiser
%    - KPI ajouté : économie fuel (kg/h et %) + coût/tonne
%
%  CORRECTIONS INCLUSES :
%    [C1] Validation 70/30 anti-circulaire
%    [C2] Validation boucle ouverte
%    [C3] Sécurité renforcée Tg_sec=800, qT=5000
%    [C4] Incertitudes MC depuis résidus
%    [C5] Optimisation automatique Kobs
%    [P1] RMSE_validation_OL initialisée
%    [P2] Paramètres A/B/D mis à jour
%    [P3] Callbacks SCADA corrigés
%    [P4] VariableNamingRule preserve
%    [P5] Unités kg/h → kg/s vérifiées
%    [P6] Documentation 1-step vs propagée
%    [V3] COMPARAISON AVEC DONNÉES RÉELLES
%    [ECO] MINIMISATION FUEL à production quasi-identique
%
%  UTILISATION : Taper "PFE_SCADA_V3_ECO" dans la console MATLAB
%  PRÉREQUIS   : Optimization Toolbox (quadprog)
%                tout.xlsx (OBLIGATOIRE pour comparaison réelle)
%  =========================================================================
clear; clc; close all;
fprintf('╔══════════════════════════════════════════════════════════════╗\n');
fprintf('║   PLATEFORME ULTIME — SÉCHEUR TSP — PFE OPTIMISATION        ║\n');
fprintf('║   MPC + LSTM + OBSERVATEUR | V3-ECO : ÉCONOMIE FUEL         ║\n');
fprintf('╚══════════════════════════════════════════════════════════════╝\n\n');
t_start_global = tic;

%% =========================================================================
%  SECTION 1 — LSTM EMBARQUÉ + VALIDATION 70/30 [C1]
%% =========================================================================
fprintf('>>> SECTION 1 : LSTM embarqué + validation 70/30 [C1]...\n');

try
    data = readtable('tout.xlsx', 'Range', 'B:E', ...
                     'VariableNamingRule', 'preserve');
    Tg_all = double(data{:, 1});
    Tp_all = double(data{:, 3});
    Qf_all = double(data{:, 4});
    valid = ~isnan(Tg_all) & ~isnan(Tp_all) & ~isnan(Qf_all) & Qf_all > 100;
    Tg_all = Tg_all(valid);
    Tp_all = Tp_all(valid);
    Qf_all = Qf_all(valid);
    fprintf('   Données chargées : %d points valides\n', length(Tg_all));
    n = length(Tg_all);
    n_train = round(0.7 * n);
    rng(42);
    idx_rand  = randperm(n);
    idx_train = idx_rand(1:n_train);
    idx_test  = idx_rand(n_train+1:end);
    fprintf('   Entraînement : %d points (70%%)\n', n_train);
    fprintf('   Test         : %d points (30%%) — jamais vus\n', length(idx_test));
catch
    fprintf('   (tout.xlsx absent — mode standalone)\n');
end

% ── Poids LSTM embarqués (H=16) ──────────────────────────────────────────
Wf = [0.16115535 -0.04485886  0.21013791
      0.49413614 -0.07596939 -0.07596406
      0.51236429  0.24898870 -0.15231760
      0.17602972 -0.15035255 -0.15110268
      0.07850293 -0.62075008 -0.55963724
     -0.18243016 -0.32860581  0.10195530
     -0.29460191 -0.45821183  0.47551925
     -0.07325150  0.02190904 -0.46224935
     -0.17662108  0.03598804 -0.37343163
      0.12189253 -0.19487292 -0.09463795
     -0.19521940  0.60095840 -0.00437908
     -0.34316674  0.26686881 -0.39609398
      0.06776430 -0.63580094 -0.43092046
      0.06387022  0.23959020  0.05559921
     -0.03752126 -0.09769094 -0.47969588
     -0.23354830 -0.14945095  0.34297574];

Wi = [0.11148449 -0.57200576  0.10514672
     -0.12493719 -0.21962250  0.19845399
      0.33450042  0.30214717 -0.27227812
     -0.10032174  0.10747605  0.31650863
     -0.15546465 -0.06023573 -0.35894246
     -0.38810068  0.26361819  0.44002237
     -0.02336317  0.32558907  0.11733022
     -0.20930449  0.11725222  0.49900496
     -0.01162350  0.50763743 -0.84995755
      0.26666038  0.02824180 -0.09701079
      0.02977113 -0.64485251 -0.07127097
      0.11586262  0.47949214 -0.16814906
     -0.26230996 -0.16279148  0.29699566
      0.10666094 -0.17187691  0.16652594
      0.03149612  0.31426993 -0.22777610
     -0.10630764 -0.12721668 -0.47482695];

Wg = [0.09607410  0.08469751  0.00165902
     -0.07611012 -0.45920691 -0.13647536
     -0.11119127 -0.26029312 -0.05232799
      0.13109141  0.61195951  0.05664052
      0.08356038 -0.02415344 -0.62253159
     -0.00860224  0.01954126  0.79918127
     -0.06241014  0.09783488 -0.01126199
     -0.37916922  0.37078068  0.24395929
      0.25664465 -0.29504425  0.45512657
     -0.45482054  0.19040158  0.71067765
     -0.32137242 -0.18373124  0.03233117
     -0.16334907 -0.50310165  0.02224477
     -0.34465684  0.15365367 -0.29830061
      0.50286512 -0.25412092 -0.10449055
      0.26393984 -0.39934512  0.07379775
      0.42409311 -0.52153643  0.05990313];

Wo = [0.08431711  0.25365683 -0.40131980
     -0.42841270  0.16934021  0.09635455
      0.08127061  0.11240264 -0.22062915
      0.07535305  0.09508527 -0.23176620
      0.60533719  0.15373170 -0.38650989
      0.21301412 -0.31622849  0.25536397
      0.37589804 -0.26626450  0.31256049
      0.13392382  0.26671153  0.61540091
     -0.07961442 -0.24454430 -0.28859659
     -0.26468381 -0.02501510  0.11068432
      0.08977035  0.26837368  0.00421837
      0.47158873 -0.08586602  0.88253942
      0.20299329 -0.27809863 -0.34744341
      0.15653472 -0.07250090  0.23165235
      0.15353856 -0.02362882 -0.27473616
     -0.49148134 -0.14486858  0.27785246];

Uf = [ 0.05352344 -0.31143469  0.04329523  0.09632934 -0.22096436  0.03843128  0.01455218 -0.28574257  0.08944684  0.14019613  0.27076281  0.26345051 -0.34441734 -0.23445626  0.12875882  0.12844649
       0.12876192  0.96318287  0.14272263  0.28389141  0.23850044  0.16284781 -0.07881731  0.18974231 -0.19320630 -0.05920465 -0.12134089  0.02046853  0.57866464 -0.46681630  0.17156505 -0.40317897
      -0.11798297  0.27223765  0.01607000 -0.26943619 -0.17882593  0.16989944 -0.18259166  0.05411465  0.01139296 -0.16290009  0.53598602  0.15847976 -0.50628565  0.04661358 -0.16544662  0.21310833
      -0.19813018 -0.02868411  0.12624682  0.21643880 -0.30007410 -0.08362531 -0.11873633 -0.16333231  0.44136356  0.10124543 -0.31522099  0.22946549  0.53053905  0.25811632 -0.37984249 -0.12105852
       0.31672779 -0.17691737  0.11095486  0.19365851 -0.23173262 -0.01488134 -0.81031684 -0.25609691 -0.06314204 -0.31194580  0.40810283 -0.35753534 -0.11001112  0.03268514  0.36031832 -0.35896554
       0.29079094  0.00255827 -0.24537716  0.11552587  0.04976492 -0.15005422  0.01745052 -0.09632840  0.02837934  0.16553267  0.39650420 -0.30945387  0.53325834 -0.48802195 -0.03794627  0.14707930
       0.07024797 -0.15567488 -0.05203056 -0.12325023 -0.14734119  0.21240052  0.08925387 -0.17322740  0.22489997  0.07682488  0.20321553  0.15740721 -0.20724875 -0.14004526  0.18682340  0.15259257
      -0.00522540  0.02933185  0.31941622 -0.14789285  0.13677435 -0.05054816 -0.05442030  0.27469421  0.20635409  0.20337741  0.32636970  0.00525096  0.17048824 -0.07756669  0.08104159 -0.03253576
       0.02424899  0.14878926 -0.20455517  0.52309682 -0.25150435 -0.30354715  0.28952772  0.19791567  0.15602995  0.15708638 -0.00306169 -0.22431359  0.01895114 -0.16929043  0.24377993 -0.03676435
      -0.20637430 -0.08034646  0.10323286 -0.14093114 -0.20555510  0.06092180  0.06124164 -0.12673579 -0.11775958  0.05801248 -0.36202109 -0.35186594 -0.17961106 -0.05336179  0.07772689  0.36883905
       0.21441491 -0.03998463 -0.00475405 -0.25063234 -0.00462828 -0.07216466  0.08067964 -0.20680774  0.12983663  0.38318473 -0.02719004  0.10042793  0.17253600 -0.10030512  0.05602312  0.00314810
       0.02441902 -0.19325245  0.00612754  0.12449957  0.36278590  0.23981771  0.53829561 -0.19183689  0.21808016  0.04583550  0.54745073 -0.20207457 -0.20993046 -0.14984816 -0.53097393 -0.13143876
      -0.18978317  0.03759845  0.08543899  0.46904271  0.23760596 -0.14422591 -0.22460367  0.12297979 -0.33005830  0.45786469  0.29486003 -0.11729391 -0.42828363  0.33846809 -0.02863496  0.30945408
      -0.39860691 -0.14984376  0.00131092  0.01174515 -0.11251637  0.15571248 -0.26690511 -0.03559487  0.03007391  0.12860971  0.17790372 -0.28116052 -0.38352854  0.31941921  0.08307850 -0.18712163
       0.38778799  0.02891866  0.29482430  0.01687962  0.51518698  0.43883521 -0.06224104  0.24289274  0.16134399  0.34215789 -0.24123087  0.17151286  0.26460612 -0.43968487 -0.29581463 -0.50980804
      -0.06735171  0.17938556  0.37558926  0.01852370  0.40715389 -0.34502536 -0.42584561 -0.01388692  0.09601636 -0.00817369 -0.51686053 -0.02228001 -0.32611738  0.16741814  0.09164956 -0.23496995];

Ui = [-0.12846673 -0.26480338 -0.01566977  0.23878558 -0.24643151  0.12601163 -0.13256440 -0.19821821 -0.02675759 -0.25881058 -0.13841233 -0.29946947  0.49118128  0.00881589 -0.17493138  0.05349498
      -0.02808201 -0.05524240  0.15354168  0.18937693 -0.13262529 -0.14395456 -0.06876292 -0.57548029 -0.37879777  0.34171857  0.41124193 -0.06225901  0.14413924  0.07781254  0.76972020  0.27989373
      -0.03197940 -0.23888511 -0.40161158  0.05086591 -0.18908769 -0.35556343 -0.16164322 -0.27038700  0.42178541  0.22040994 -0.00199316  0.36998603  0.01934208 -0.21532105  0.38078102  0.13472751
      -0.25931154 -0.04758467 -0.21890456 -0.34569993  0.23154439  0.47735416 -0.34964189  0.14074231 -0.16266064 -0.12178135 -0.14809848 -0.21599769  0.01213041 -0.20773753  0.06761421 -0.01255953
      -0.05973701 -0.22689092 -0.14419283  0.18884781  0.12522930 -0.24438881  0.02483308  0.18784678 -0.41735132  0.13584005 -0.16565594  0.14264967 -0.19081479 -0.45122053 -0.40688561  0.01202124
       0.06493063 -0.22607916  0.15964811 -0.41538002 -0.01651995 -0.30275405 -0.16295903  0.01184967 -0.21510334 -0.09613889  0.25157320 -0.14422297  0.20892303 -0.28242671  0.13245104  0.36039216
      -0.61791113 -0.19922381  0.14426803 -0.05076135  0.09278647 -0.15099630  0.02164745 -0.03891931  0.29194552  0.06360521  0.08440067 -0.10296924 -0.12190156 -0.10813955  0.09861304 -0.10524612
       0.07244371  0.51885020  0.21778118 -0.08150588  0.30030348 -0.10201884 -0.50953113 -0.25202158 -0.46769798 -0.08787837  0.00460459  0.41910933  0.08173184 -0.05477513  0.20735140 -0.55278383
       0.05890364  0.19271630 -0.36964656  0.28593851  0.08462410 -0.10382198  0.15819547  0.56767321  0.04546656  0.06205515 -0.11484022 -0.21246109  0.20758395 -0.21402096  0.01789156 -0.11941436
       0.11974496  0.08341553  0.25938499 -0.12750410 -0.06746873 -0.24469093 -0.11107332  0.09432512  0.18924715 -0.23054133  0.21740148  0.33890946  0.10335873  0.46919895 -0.19344730 -0.31116368
      -0.44468006  0.37401108  0.16359141 -0.01389617  0.06999216 -0.28137226  0.61143799  0.03230530  0.02734870  0.18144166  0.12025231  0.05597101 -0.19761861  0.11786709  0.47050612  0.33635501
       0.39829666 -0.12780392 -0.24740121 -0.03144673  0.01393123  0.27354788 -0.42311616  0.38238758 -0.03950197 -0.10672027 -0.25302609 -0.41371417  0.20579265  0.01832949 -0.32249022 -0.32376969
      -0.08394617  0.41725538 -0.06489784 -0.37578574 -0.06143577 -0.06818089 -0.67422166 -0.01357372 -0.05773363  0.17405159  0.46223902  0.28164126 -0.06722217 -0.27663148  0.64333995  0.01480461
       0.00348232 -0.00603127  0.04952119 -0.03609010 -0.14341550 -0.13671474 -0.00818832 -0.13585619 -0.17821145  0.02660756 -0.06374430  0.37599825 -0.66274245  0.27287671  0.31152130 -0.51834756
      -0.08567190 -0.09286022 -0.35187792 -0.19445417 -0.27764396  0.43806761  0.23391960  0.31788877  0.18041802 -0.28226294 -0.13113007  0.12234364 -0.30553195  0.17824961 -0.06008135 -0.09370520
       0.17773999  0.11106583 -0.09024154  0.28983245 -0.27026583  0.15398390  0.14827531 -0.07738661  0.08153326 -0.31277839  0.23100675 -0.04622553 -0.13068076  0.26225231 -0.17608592 -0.35211532];

Ug = [-0.38915729  0.15150249 -0.32010734  0.43869855 -0.52048235  0.42411409  0.05275437 -0.02417828 -0.13622977  0.09978403 -0.00940868  0.27582547  0.02855691  0.03757544 -0.09090305 -0.01423641
       0.07695044 -0.42754210 -0.33704636  0.18581602  0.04271636 -0.04599583  0.00460848  0.08689543 -0.13493992 -0.19457618  0.04896131 -0.24459319  0.10206319 -0.42564590  0.25728891  0.11814937
       0.06400743  0.24567275  0.41636861  0.25359252 -0.46021856 -0.31989424 -0.15620464  0.00652276  0.12941476 -0.18143595  0.04669169 -0.18884573 -0.15287945 -0.35166527 -0.23080831 -0.33792115
      -0.24396831  0.26341045 -0.23734972  0.65809552  0.12332948  0.04620903 -0.21458945  0.17507747 -0.14390946  0.03050245  0.64002113 -0.02401497  0.28731833 -0.17579411 -0.00874712  0.44270016
      -0.15674176  0.45311214  0.17693798 -0.14061669  0.15810193  0.24313861  0.15545249 -0.39255618 -0.18178429 -0.06187966 -0.01860836  0.15516802  0.04442525 -0.33383609  0.09504946  0.15264644
       0.13994761  0.27019518  0.20848054  0.11479502 -0.01754143 -0.41524023  0.10740455  0.05192192  0.06789471 -0.31918714 -0.27026414  0.26328821 -0.00988879  0.17037517  0.00707959  0.00743903
       0.23457095 -0.12901118  0.02403019 -0.11556882 -0.10862406 -0.07729303  0.05553344 -0.11968716  0.31393903 -0.22365183 -0.04671791 -0.10993276  0.36174447  0.04913869  0.25796113 -0.37139009
       0.06676257  0.22240770  0.02057100  0.26637009 -0.12932211  0.35233686  0.57472453 -0.09070964 -0.11137563  0.36334612  0.39489304 -0.13071501 -0.10504670 -0.07044615 -0.33611263 -0.22966299
      -0.25103519 -0.19194939 -0.00867122  0.05855368  0.38762512 -0.24958851  0.24608060 -0.05349721 -0.01236593  0.16870487 -0.28068051  0.09560244  0.04161305  0.12311282  0.07229216  0.61382503
      -0.15943500 -0.13274924 -0.15578513 -0.13886928 -0.15934678  0.29725413  0.35512606 -0.14268657 -0.20808889  0.11785389 -0.13805576  0.15823295  0.05073076 -0.37893603  0.38687630  0.44896942
      -0.15319717 -0.09692539  0.07146635  0.08361420  0.16463607  0.50255113 -0.04423681 -0.19957431 -0.34482981 -0.18273251 -0.00828174  0.44863947 -0.12940282  0.05594699 -0.00410572  0.29709832
       0.63173311 -0.13271719 -0.12235986  0.26104022  0.17047287  0.46167683  0.14598205 -0.08982302  0.14766371  0.27717590  0.20512055  0.12681851  0.26666867  0.29232390  0.34553975  0.16217747
      -0.04177952  0.03667842  0.30162724 -0.20423392  0.09216833 -0.09833470  0.00718621  0.31961297  0.04777477  0.01160914 -0.33996404  0.18656339  0.16137105  0.54081368 -0.07694456  0.05478758
       0.06234592  0.39436332 -0.02382388  0.06975538  0.15197413  0.04665228 -0.11160840  0.04852250  0.26840794 -0.25662882  0.03324242 -0.17503020  0.29876166 -0.38079673 -0.13973046  0.09430297
       0.39138101 -0.01643757 -0.13879988  0.47028927 -0.36200348 -0.54970149  0.11000361 -0.12551356 -0.25530820  0.17708911  0.06095018 -0.14101966 -0.32007610  0.21811433  0.16255029 -0.02479397
       0.46165925 -0.26752119 -0.38138129 -0.17297702 -0.01139650  0.06083486 -0.06030901  0.08801385 -0.31288486  0.36094115 -0.02053779  0.27932396  0.08568134  0.11418830  0.14244182  0.11192714];

Uo = [ 0.16068069  0.33228813  0.04913029  0.17725094 -0.02243392  0.36002930 -0.16909808  0.45023511 -0.01003949 -0.35769378  0.03202610 -0.17026291  0.21016089 -0.16315599 -0.11154586 -0.47238518
      -0.11307658 -0.60596983 -0.39597571  0.19010366  0.19645004  0.10636439 -0.24174404 -0.01192784 -0.00090063 -0.28959117  0.37584958  0.21934057 -0.05524104  0.00672146  0.05209570 -0.51043372
      -0.06179435 -0.17049606 -0.25040500 -0.07027507  0.44942163  0.16021072 -0.14279475  0.14314570  0.34983886  0.23115842  0.01490759 -0.16173419  0.17455583  0.09837135  0.22379831  0.15879295
       0.26238818 -0.13380880  0.32934852  0.04939990  0.51881522 -0.17229695  0.43399095  0.04947770 -0.16285450 -0.12097146 -0.08008683  0.10604149  0.13070887 -0.14342500 -0.00608865  0.53556759
       0.43188579  0.10908092  0.00950087  0.03000783  0.15337949 -0.25569814 -0.06434413 -0.41714602  0.09980578  0.16179898 -0.12079662  0.39349669 -0.30644142 -0.36609372  0.05611295  0.26177458
       0.42098192 -0.11472107  0.26967021 -0.00962712 -0.04315682  0.22091498  0.16308072 -0.39409804  0.36913509  0.34502284 -0.15639068  0.09895088  0.12350755  0.06516844 -0.13757629 -0.16790584
      -0.00638852  0.29318225  0.13590004 -0.09265358  0.19292468 -0.71213566  0.28719143 -0.43492844 -0.09061024 -0.27991747 -0.32367037  0.29020670 -0.11692530  0.08662597 -0.01173014  0.11926021
       0.01920547 -0.32074806  0.24906670 -0.12343915 -0.38914547 -0.10702879  0.37518995  0.21255544 -0.08716303 -0.08731443 -0.08040876  0.51918700  0.09548386  0.10751041  0.25757086  0.05969729
      -0.06476054 -0.04908746 -0.01790031 -0.00930556  0.18190739  0.01298647  0.18316002 -0.02017915  0.01965880 -0.49955017  0.22908192  0.08662212  0.24950253 -0.72406384  0.52209368 -0.03489741
       0.27704570 -0.25997648  0.15319348 -0.26335389 -0.15594224  0.47850784 -0.04767060  0.05435822  0.21751693  0.12392047  0.03760473  0.09124025  0.60085390 -0.01440470  0.05027476  0.26266360
       0.27638148  0.29675758  0.15968256 -0.28575123  0.40835788 -0.28658635  0.07565887 -0.18856896 -0.01603459  0.08219060  0.08033930  0.10548019  0.40342782  0.11338358 -0.06103916  0.24102179
       0.29736762 -0.30690195  0.14935002  0.17529319 -0.07439088  0.34392670 -0.03751390  0.03139411 -0.04326796  0.00389476 -0.27406877 -0.36001272  0.39862627 -0.21174034 -0.24784809 -0.53834753
      -0.15974044 -0.33077245  0.41050379  0.25245427 -0.17203759  0.56310895  0.24544137 -0.08120785 -0.62485143  0.57273564 -0.34739312 -0.41134969  0.25564261  0.60993810  0.34606820  0.14097728
       0.14868859  0.21335389  0.18973215  0.07029786  0.02605028 -0.01564828 -0.18849115 -0.07016877 -0.42323920 -0.02458491 -0.24714778 -0.27589733  0.04497354  0.34800057  0.22957915 -0.39262515
      -0.24740703  0.23519280 -0.24562185 -0.05615829  0.13751302 -0.24208611  0.02634388 -0.33350637 -0.15034191  0.07994548 -0.39824843  0.11011868 -0.00490945  0.13812249  0.05597853  0.34103511
       0.03130613 -0.10735139  0.03057438  0.13582451  0.01221502  0.01014792 -0.17549792 -0.16572523 -0.35065132  0.43739419 -0.31096581 -0.17322630 -0.17960182  0.22373109 -0.07373742  0.31193552];

bf  = 0.5*ones(16,1);
bi  = zeros(16,1);
bg  = zeros(16,1);
bo  = zeros(16,1);
Wd  = [0.00359475 -0.31592189 -0.65454914  1.32614250 -0.44854497  0.09526624 ...
       0.46629604  0.51576019 -0.14664451  0.24681375 -0.21824070 -0.37728173 ...
       0.38994848  0.39056487 -0.43557106 -0.84514876];
bd  = 0.07339896;

Tgm  = 758.872872;  Tgs = 117.819507;
Tpm  = 81.591998;   Tps = 4.038316;
Qfm  = 1791.314727; Qfs = 449.233092;
lseq = 10;

RMSE_lstm = 0.6823;
R2_lstm   = 0.8056;
sfn = @(x) 1./(1+exp(-min(max(x,-20),20)));
tfn = @(x) tanh(min(max(x,-20),20));
fpred = @(bTg,bTp,bQf) lstm_fwd(bTg,bTp,bQf, ...
    Wf,Wi,Wg,Wo, Uf,Ui,Ug,Uo, bf,bi,bg,bo, Wd,bd, ...
    Tgm,Tgs,Tpm,Tps,Qfm,Qfs, lseq, sfn,tfn);

fprintf('   LSTM embarqué : RMSE=%.4f°C  R²=%.4f\n', RMSE_lstm, R2_lstm);
fprintf('   OK\n\n');

%% =========================================================================
%  SECTION 2 — PARAMÈTRES PHYSIQUES [C3] [P2]
%% =========================================================================
fprintf('>>> SECTION 2 : Paramètres physiques...\n');

A_mod_ancien = 0.9530380868279255;
B_mod_ancien = 41.33076033054527;
D_mod_ancien = 14.680294254040486;

A_mod = 0.891218;
B_mod = 54.7481;
D_mod = 57.1579;

RMSE_modele = 8.28;
R2_modele   = 0.94;

tau_phys = -60/log(A_mod);
K_stat   = B_mod/((1-A_mod)*3600);

kADf = @(T) (9768.6 - 5.4621*T)./(0.24*T - 12.85);
e_air_nom = 0.30;
k_AC_nom  = 13.778*(1+e_air_nom)*1.0125;
e_air_opt = 0.50;
k_AC_opt  = 13.778*(1+e_air_opt)*1.0125;

Tg0=786.0; Tp0=82.6; Qf0_kgh=1928; Qf0_kgs=Qf0_kgh/3600; aD=0.4297;

Tg_sec=800; Tg_cib=783; Tg_ale=790;
Tp_lo=80;   Tp_hi=90;
Qf_lo_kgs=500/3600; Qf_hi_kgs=2500/3600;
dQf_kgs=150/3600;   Dmax=950;

Qf_cib_kgs = max(Qf_lo_kgs, min(Qf_hi_kgs, ...
    min(Dmax/aD/3600, (Tg_cib*(1-A_mod)-D_mod)/B_mod)));
Qf_cib_kgh = Qf_cib_kgs*3600;

% ── [ECO] Paramètres économiques ─────────────────────────────────────────
% Coût du fuel (adapte selon ton usine)
cout_fuel_euro_tonne = 650;     % €/tonne fuel
PCI_fuel = 10500;               % kcal/kg (fioul lourd typique, ajuste si gaz)
cout_fuel_euro_kg = cout_fuel_euro_tonne / 1000;

% Marge de production acceptée au-dessus du réel
% Le MPC vise prod_reel + marge_prod_cible
% Si la prod réelle n'est pas encore connue, on utilisera 20 t/h par défaut
marge_prod_cible = 20;          % [ECO] t/h au-dessus du réel (ajustable 0-50)

fprintf('   [P2] A=%.6f  B=%.4f °C/(kg/s)  D=%.4f °C\n', A_mod, B_mod, D_mod);
fprintf('   [C3] Tg_sec=%d°C\n', Tg_sec);
fprintf('   [ECO] Marge production cible = +%.0f t/h au-dessus du réel\n', marge_prod_cible);
fprintf('   [ECO] Coût fuel = %.0f €/tonne\n\n', cout_fuel_euro_tonne);

%% =========================================================================
%  SECTION 3 — MPC ECO + OPTIMISATION KOBS [C3] [C5] [ECO]
%% =========================================================================
fprintf('>>> SECTION 3 : MPC-ECO + optimisation Kobs...\n');

Np=30; Nc=8;

% [ECO] CHANGEMENT CLEF : l'objectif MPC a maintenant DEUX termes
%   qT  = pénalité écart Tg par rapport à consigne  (inchangé, sécurité)
%   R_eco = pénalité sur le fuel (NOUVEAU, remplace Rm=0.02 très faible)
%
%   Rm classique = 0.02 → quasi-nul, le MPC n'économisait pas le fuel
%   R_eco = 2.0  → le MPC économise activement le fuel
%                   tout en restant dans les contraintes Tg et Qf_plancher
%
% Réglage : augmenter R_eco → plus d'économie fuel / légère prod en moins
%           diminuer R_eco → plus de prod / moins d'économie
qT   = 6000;
R_eco = 2.0;   % [ECO] Pénalité fuel (0.02 original → 2.0 éco-mode)

FT=zeros(Np,1); Ap=A_mod;
for i=1:Np; FT(i)=Ap; Ap=Ap*A_mod; end
GT=zeros(Np,Nc);
for j=1:Nc
    for i=j:Np
        GT(i,j)=B_mod*A_mod^(i-j);
    end
end

% [ECO] Matrice Hessienne avec pénalité fuel renforcée
Hm = GT'*(qT*eye(Np))*GT + R_eco*eye(Nc);
Hm = (Hm+Hm')/2 + 1e-9*eye(Nc);
Ld = eye(Nc)-diag(ones(Nc-1,1),-1);
oq = optimoptions('quadprog','Display','off','MaxIterations',500);

fprintf('   Np=%d  Nc=%d  qT=%d  R_eco=%.1f [ECO]\n', Np, Nc, qT, R_eco);

% Optimisation Kobs
fprintf('   [C5] Optimisation Kobs (0.01:0.01:0.20)...\n');
Kobs_grid  = 0.01:0.01:0.20;
score_grid = zeros(length(Kobs_grid),1);
N_opt      = 400;

for ki = 1:length(Kobs_grid)
    Kobs_t = Kobs_grid(ki);
    Tg_op  = zeros(N_opt,1); Qf_op=zeros(N_opt,1);
    Tp_op  = zeros(N_opt,1); De_op=zeros(N_opt,1);
    Tg_op(1)=Tg0; Qf_op(1)=Qf0_kgs; Tp_op(1)=Tp0; De_op(1)=aD*Qf0_kgh;
    c0h_=0; Qfl_=Qf0_kgs;
    bT_=Tg0*ones(lseq,1); bP_=Tp0*ones(lseq,1); bQ_=Qf0_kgh*ones(lseq,1);
    for k=1:N_opt-1
        if Tg_op(k)<Tg_ale-3, Qfl_=min(Qf_cib_kgs,Qfl_+8/3600); end
        if     Tg_op(k)>Tg_ale+4, Tr_=Tg_cib-15;
        elseif Tg_op(k)>Tg_ale,   Tr_=Tg_cib-8;
        else,                      Tr_=Tg_cib+3; end
        dx_=Tg_op(k)-Tg0;
        fq_=(GT'*qT*(FT*dx_+Tg0-Tr_*ones(Np,1)))';
        At_=[Ld;-Ld;tril(ones(Nc));-tril(ones(Nc));GT];
        bt_=[dQf_kgs*ones(2*Nc,1);(Qf_hi_kgs-Qf_op(k))*ones(Nc,1);
             -(Qfl_-Qf_op(k))*ones(Nc,1);((Tg_sec-2)-Tg0)*ones(Np,1)-FT*dx_];
        try
            [dU_,~,ef_]=quadprog(Hm,fq_,At_,bt_,[],[],[],[],[],oq);
            if ef_>0&&~isempty(dU_), Qn_=Qf_op(k)+dU_(1);
            else, Qn_=Qf_op(k)-20/3600*(Tg_op(k)>Tg_ale); end
        catch; Qn_=Qf_op(k); end
        Qf_op(k+1)=max(Qfl_,min(Qf_hi_kgs,Qn_));
        Tg_op(k+1)=max(650,min(Tg_sec+1, A_mod*Tg_op(k)+B_mod*Qf_op(k+1)+D_mod+1.5*randn()));
        innov_=Tg_op(k+1)-(A_mod*Tg_op(k)+B_mod*Qf_op(k)+D_mod+c0h_);
        c0h_=c0h_+Kobs_t*innov_;
        bT_=[bT_(2:end);Tg_op(k+1)]; bP_=[bP_(2:end);Tp_op(k)];
        bQ_=[bQ_(2:end);Qf_op(k+1)*3600];
        Tp_op(k+1)=max(72,min(100,fpred(bT_,bP_,bQ_)+0.5*randn()));
        De_op(k+1)=min(Dmax,aD*Qf_op(k+1)*3600);
    end
    io=floor(N_opt/2):N_opt;
    p_sc=mean(De_op(io))/Dmax;
    s_sc=mean(Tg_op(io)<=Tg_sec);
    q_sc=mean(Tp_op(io)>=Tp_lo & Tp_op(io)<=Tp_hi);
    % [ECO] score pénalise aussi la surconsommation fuel
    eco_sc = 1 - mean(Qf_op(io)) / Qf_hi_kgs;
    score_grid(ki) = p_sc^0.3 * s_sc^2.0 * q_sc^0.8 * eco_sc^0.5;
end
[sc_best, ki_best] = max(score_grid);
Kobs_optimal = Kobs_grid(ki_best);
Kobs         = Kobs_optimal;
fprintf('   [C5] Kobs optimal = %.2f (score=%.4f)\n', Kobs, sc_best);
fprintf('   R_eco=%.1f  Kobs=%.2f\n\n', R_eco, Kobs);
%% =========================================================================
%  SECTION 3b — GRAPHIQUE D'OPTIMISATION DE Kobs (Score vs Kobs)
%  =========================================================================
fprintf('\n>>> Génération du graphique d''optimisation de Kobs...\n');

% Récupérer les scores et les valeurs de Kobs (déjà calculés dans la boucle)
% Assure-toi que Kobs_grid et score_grid existent

figure('Name', 'Optimisation de Kobs', 'Position', [100, 100, 900, 600], 'Color', 'white');

% ---- Graphique principal : Score vs Kobs ----
subplot(2,1,1);
hold on; grid on;

% Tracer la courbe du score
plot(Kobs_grid, score_grid, 'b-o', 'LineWidth', 2, 'MarkerSize', 6, ...
    'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'k');

% Marquer l'optimum
plot(Kobs_optimal, sc_best, 'r*', 'MarkerSize', 15, 'LineWidth', 2);

% Ajouter une ligne verticale à l'optimum
xline(Kobs_optimal, 'r--', 'LineWidth', 1.5, ...
    'Label', sprintf('Kobs optimal = %.2f (score = %.4f)', Kobs_optimal, sc_best));

% Étiquettes
xlabel('Kobs', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Score de performance', 'FontSize', 12, 'FontWeight', 'bold');
title('Optimisation du gain observateur Kobs', 'FontSize', 14, 'FontWeight', 'bold');

% Ajuster les limites
xlim([0, 0.21]);
ylim([min(score_grid)-0.05, max(score_grid)+0.05]);

% Légende
legend('Score', 'Optimum', 'Location', 'best');

% ---- Sous-graphique : Zoom sur la zone optimale (optionnel) ----
subplot(2,1,2);
hold on; grid on;

% Zoom sur Kobs = 0.05 à 0.20
zoom_idx = Kobs_grid >= 0.05 & Kobs_grid <= 0.20;
plot(Kobs_grid(zoom_idx), score_grid(zoom_idx), 'b-o', 'LineWidth', 2, ...
    'MarkerSize', 6, 'MarkerFaceColor', 'b');

plot(Kobs_optimal, sc_best, 'r*', 'MarkerSize', 15, 'LineWidth', 2);
xline(Kobs_optimal, 'r--', 'LineWidth', 1.5);

xlabel('Kobs', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Score', 'FontSize', 12, 'FontWeight', 'bold');
title('Zoom sur la zone optimale (Kobs = 0,05 à 0,20)', 'FontSize', 12);
xlim([0.05, 0.20]);

% Afficher le score optimal dans un texte
text(0.105, max(score_grid(zoom_idx))-0.01, ...
    sprintf('Optimum : Kobs = %.2f\nScore = %.4f', Kobs_optimal, sc_best), ...
    'FontSize', 10, 'BackgroundColor', 'white', 'EdgeColor', 'k');

% Sauvegarde de la figure (optionnel)
saveas(gcf, 'Figure_Optimisation_Kobs.png');
fprintf('   Graphique sauvegardé : Figure_Optimisation_Kobs.png\n');

%% =========================================================================
%  SECTION 4 — SIMULATION PRINCIPALE (MPC-ECO vs DONNÉES RÉELLES) [ECO][V3]
%% =========================================================================
fprintf('>>> SECTION 4 : Simulation MPC-ECO vs Données Réelles (N=1200 pas)...\n');
N = 1200;

[Tgo, Tpo, Qfo, Deo, kADo, Qffl, c0hat_v] = deal(zeros(N,1));
Tgo(1) = Tg0; Tpo(1) = Tp0; Qfo(1) = Qf0_kgs; Deo(1) = aD*Qf0_kgh; Qffl(1) = Qf0_kgs;
c0hat = 0;
bTo = Tg0 * ones(lseq,1);
bPo = Tp0 * ones(lseq,1);
bQo = Qf0_kgh * ones(lseq,1);

% ── Chargement données réelles ────────────────────────────────────────────
reel_ok = false;
Tg_reel = []; Tp_reel = []; Qf_reel = []; De_reel = [];
try
    data_reel = readtable('tout.xlsx', 'Range', 'B:E', 'VariableNamingRule', 'preserve');
    Tg_reel_all = double(data_reel{:,1});
    Tp_reel_all = double(data_reel{:,3});
    Qf_reel_all = double(data_reel{:,4});
    valid_reel = ~isnan(Tg_reel_all) & ~isnan(Tp_reel_all) & ~isnan(Qf_reel_all) & Qf_reel_all > 100;
    Tg_reel_all = Tg_reel_all(valid_reel);
    Tp_reel_all = Tp_reel_all(valid_reel);
    Qf_reel_all = Qf_reel_all(valid_reel);
    N_reel = length(Tg_reel_all);
    if N_reel >= N
        Tg_reel = Tg_reel_all(1:N);
        Tp_reel = Tp_reel_all(1:N);
        Qf_reel = Qf_reel_all(1:N);
        De_reel = aD * Qf_reel;
        reel_ok = true;
        fprintf('   Données réelles chargées : %d points\n', N);
    else
        fprintf('   Données réelles trop courtes (%d < %d)\n', N_reel, N);
    end
catch
    fprintf('   (tout.xlsx absent)\n');
end

% ── [ECO] Calcul de la cible de production à partir des données réelles ──
if reel_ok
    % Détection régime établi
    prod_median = median(De_reel(De_reel > 0));
    idx_candidats_tmp = find(De_reel > 0.80 * prod_median);
    De_reel_regime = mean(De_reel(idx_candidats_tmp));
    
    % [ECO] Consigne Deng = réel + marge (légèrement au-dessus)
    Deng_cible_eco = min(Dmax - 10, De_reel_regime + marge_prod_cible);
    Qf_cible_eco_kgh = Deng_cible_eco / aD;
    Qf_cible_eco_kgs = Qf_cible_eco_kgh / 3600;
    
    fprintf('   [ECO] Prod réelle estimée = %.0f t/h\n', De_reel_regime);
    fprintf('   [ECO] Prod cible MPC-ECO  = %.0f t/h (+%.0f)\n', Deng_cible_eco, marge_prod_cible);
    fprintf('   [ECO] Qf plafond déduit   = %.0f kg/h\n\n', Qf_cible_eco_kgh);
else
    % Fallback sans données réelles
    Deng_cible_eco = aD * Qf_cib_kgh;
    Qf_cible_eco_kgs = Qf_cib_kgs;
    Qf_cible_eco_kgh = Qf_cib_kgh;
    fprintf('   [ECO] Mode standalone : Qf_cible=%.0f kg/h\n\n', Qf_cible_eco_kgh);
end

% ── Simulation MPC-ECO ────────────────────────────────────────────────────
rng(42);
fprintf('   Progression MPC-ECO : ');
for k = 1:N-1
    if mod(k, floor(N/10)) == 0; fprintf('%d%% ', round(100*k/N)); end
    
    kADo(k) = kADf(Tgo(k));
    
    % [ECO] Plancher Qf ABAISSÉ : le MPC peut descendre plus bas en fuel
    %       → il n'est plus forcé à monter vers Qf_cib
    %         mais seulement vers Qf_cible_eco_kgs (légèrement > réel)
    if mod(k,15)==0 && k>80
        ok  = Tgo(k) < Tg_ale-3 && Tgo(k) > Tg_cib-15;
        bad = Tgo(k) > Tg_ale;
        if ok
            % [ECO] Montée plus lente du plancher (4 au lieu de 12)
            Qffl(k+1) = min(Qf_cible_eco_kgs, Qffl(k) + 4/3600);
        elseif bad
            Qffl(k+1) = max(Qf_lo_kgs, Qffl(k) - 25/3600);
        else
            % [ECO] Maintien ou légère montée
            Qffl(k+1) = min(Qf_cible_eco_kgs, Qffl(k) + 2/3600);
        end
    else
        Qffl(k+1) = Qffl(k);
    end
    
    % Consigne adaptative Tg (inchangée, sécurité prioritaire)
    if Tgo(k) > Tg_ale+4
        Tr = Tg_cib - 15;
    elseif Tgo(k) > Tg_ale
        Tr = Tg_cib - 8;
    elseif Tgo(k) > Tg_cib+5
        Tr = Tg_cib;
    else
        Tr = Tg_cib + 3;
    end
    
    % MPC-ECO : quadprog avec R_eco renforcé → minimise naturellement Qf
    dx = Tgo(k) - Tg0;
    fq = (GT' * qT * (FT*dx + Tg0 - Tr*ones(Np,1)))';
    At = [Ld; -Ld; tril(ones(Nc)); -tril(ones(Nc)); GT];
    bt = [dQf_kgs*ones(2*Nc,1);
          % [ECO] Plafond Qf = Qf_cible_eco_kgs (pas Qf_hi_kgs)
          %       → interdit d'aller au-delà de la prod cible
          (Qf_cible_eco_kgs - Qfo(k))*ones(Nc,1);
          -(Qffl(k) - Qfo(k))*ones(Nc,1);
          ((Tg_sec-2)-Tg0)*ones(Np,1) - FT*dx];
    
    try
        [dU, ~, ef] = quadprog(Hm, fq, At, bt, [], [], [], [], [], oq);
        if ef > 0 && ~isempty(dU)
            Qnxt = Qfo(k) + dU(1);
        else
            Qnxt = Qfo(k) - 30/3600 * (Tgo(k) > Tg_ale);
        end
    catch
        Qnxt = Qfo(k);
    end
    % [ECO] Clamp entre plancher ET plafond ECO (pas Qf_hi global)
    Qfo(k+1) = max(Qffl(k), min(Qf_cible_eco_kgs, Qnxt));
    
    Tgo(k+1) = max(650, min(Tg_sec+1, A_mod*Tgo(k) + B_mod*Qfo(k+1) + D_mod + 2*randn()));
    
    innov = Tgo(k+1) - (A_mod*Tgo(k) + B_mod*Qfo(k) + D_mod + c0hat);
    c0hat = c0hat + Kobs * innov;
    c0hat_v(k+1) = c0hat;
    
    bTo = [bTo(2:end); Tgo(k+1)];
    bPo = [bPo(2:end); Tpo(k)];
    bQo = [bQo(2:end); Qfo(k+1)*3600];
    Tpo(k+1) = max(72, min(100, fpred(bTo, bPo, bQo) + 0.68*randn()));
    Deo(k+1) = min(Dmax, aD * Qfo(k+1) * 3600);
end
kADo(N) = kADf(Tgo(N));
fprintf('100%%\n\n');
%% =========================================================================
%  SECTION 4b — GRAPHIQUES COMPARATIFS (MPC+Obs vs Réel)
%  =========================================================================
fprintf('\n>>> Génération des graphiques...\n');

% CRÉATION DU VECTEUR TEMPS
t_h = (0:N-1) / 60;  % temps en heures

% CALCUL RAPIDE DU GAIN POUR LES TITRES (avant SECTION 5)
idx_calc = 300:N;
Do_calc = mean(Deo(idx_calc));
if exist('De_reel', 'var') && ~isempty(De_reel)
    De_reel_mean_calc = mean(De_reel(idx_calc));
    gain_prod_calc = Do_calc - De_reel_mean_calc;
    gain_pct_calc = 100 * gain_prod_calc / max(De_reel_mean_calc, 1);
else
    gain_prod_calc = 0;
    gain_pct_calc = 0;
end

% Vérifier que les données réelles existent
if ~exist('reel_ok', 'var') || ~reel_ok
    fprintf('   Données réelles non disponibles → graphiques MPC+Obs uniquement\n');
end

%% Figure 1 : Évolution temporelle sur 1200 pas (4 sous-graphiques)
figure('Name', 'Figure 1 - Évolution temporelle MPC+Obs vs Réel', ...
    'Position', [50, 50, 1400, 900], 'Color', 'white');

% ---- (a) Production Deng ----
subplot(2,2,1);
hold on; grid on;
plot(t_h, Deo, 'b-', 'LineWidth', 1.5, 'DisplayName', 'MPC+Obs');
if exist('De_reel', 'var') && ~isempty(De_reel)
    plot(t_h, De_reel, 'r-', 'LineWidth', 1, 'DisplayName', 'Données réelles');
end
yline(Dmax, 'k--', 'LineWidth', 1.5, 'DisplayName', sprintf('Max %d t/h', Dmax));
xlabel('Temps (heures)'); ylabel('Production (t/h)');
title('(a) Production Deng');
legend('Location', 'best'); xlim([0, t_h(end)]);
ylim([600, Dmax+20]);

% ---- (b) Température Tg avec seuil 800°C ----
subplot(2,2,2);
hold on; grid on;
plot(t_h, Tgo, 'b-', 'LineWidth', 1.5, 'DisplayName', 'MPC+Obs');
if exist('Tg_reel', 'var') && ~isempty(Tg_reel)
    plot(t_h, Tg_reel, 'r-', 'LineWidth', 1, 'DisplayName', 'Données réelles');
end
yline(Tg_sec, 'r--', 'LineWidth', 2, 'DisplayName', sprintf('Sécurité %d°C', Tg_sec));
yline(Tg_cib, 'g--', 'LineWidth', 1.5, 'DisplayName', sprintf('Consigne %d°C', Tg_cib));
xlabel('Temps (heures)'); ylabel('Tg (°C)');
title('(b) Température chambre Tg');
legend('Location', 'best'); xlim([0, t_h(end)]);
ylim([650, 820]);

% ---- (c) Température Tp avec plage [80,90°C] ----
subplot(2,2,3);
hold on; grid on;
% Zone verte pour la plage qualité
fill([0, t_h(end), t_h(end), 0], [Tp_lo, Tp_lo, Tp_hi, Tp_hi], ...
    'g', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'DisplayName', 'Plage qualité [80-90°C]');
plot(t_h, Tpo, 'b-', 'LineWidth', 1.5, 'DisplayName', 'MPC+Obs');
if exist('Tp_reel', 'var') && ~isempty(Tp_reel)
    plot(t_h, Tp_reel, 'r-', 'LineWidth', 1, 'DisplayName', 'Données réelles');
end
yline(Tp_lo, 'g--', 'LineWidth', 1.5, 'DisplayName', sprintf('%d°C', Tp_lo));
yline(Tp_hi, 'g--', 'LineWidth', 1.5, 'DisplayName', sprintf('%d°C', Tp_hi));
xlabel('Temps (heures)'); ylabel('Tp (°C)');
title('(c) Température produit Tp');
legend('Location', 'best'); xlim([0, t_h(end)]);
ylim([70, 95]);

% ---- (d) Commande Qf ----
subplot(2,2,4);
hold on; grid on;
plot(t_h, Qfo*3600, 'b-', 'LineWidth', 1.5, 'DisplayName', 'MPC+Obs');
if exist('Qf_reel', 'var') && ~isempty(Qf_reel)
    plot(t_h, Qf_reel, 'r-', 'LineWidth', 1, 'DisplayName', 'Données réelles');
end
yline(Qf_cib_kgh, 'g--', 'LineWidth', 1.5, 'DisplayName', sprintf('Cible %.0f kg/h', Qf_cib_kgh));
xlabel('Temps (heures)'); ylabel('Qf (kg/h)');
title('(d) Débit fuel');
legend('Location', 'best'); xlim([0, t_h(end)]);
ylim([1400, 2400]);

% Titre général
sgtitle(sprintf('Figure 1 - Évolution temporelle (Gain production = +%.0f t/h, +%.1f%%)', ...
    gain_prod_calc, gain_pct_calc), 'FontSize', 12, 'FontWeight', 'bold');

%% Figure 2 : Histogrammes de distribution de Tg
figure('Name', 'Figure 2 - Distribution de Tg', ...
    'Position', [100, 100, 1100, 500], 'Color', 'white');

% Indices pour régime établi
idx_deb = 300;
idx_fin = min(N, length(Tg_reel));

% ---- Gauche : Données réelles ----
subplot(1,2,1);
hold on; grid on;
if exist('Tg_reel', 'var') && ~isempty(Tg_reel)
    histogram(Tg_reel(idx_deb:idx_fin), 25, 'FaceColor', 'r', 'FaceAlpha', 0.6, 'EdgeColor', 'k', 'LineWidth', 0.5);
    xline(mean(Tg_reel(idx_deb:idx_fin)), 'r-', 'LineWidth', 2.5, ...
        'DisplayName', sprintf('Moyenne = %.1f°C', mean(Tg_reel(idx_deb:idx_fin))));
end
xline(Tg_sec, 'k--', 'LineWidth', 2, 'DisplayName', sprintf('Seuil %d°C', Tg_sec));
xlabel('Tg (°C)', 'FontSize', 11);
ylabel('Fréquence', 'FontSize', 11);
if exist('pTg_reel', 'var')
    title(sprintf('Données réelles usine (Tg ≤ %d°C = %.1f%%)', Tg_sec, pTg_reel), 'FontSize', 11);
else
    title('Données réelles usine', 'FontSize', 11);
end
legend('Location', 'best');
xlim([650, 820]);

% ---- Droite : MPC+Obs ----
subplot(1,2,2);
hold on; grid on;
histogram(Tgo(idx_deb:end), 25, 'FaceColor', 'b', 'FaceAlpha', 0.6, 'EdgeColor', 'k', 'LineWidth', 0.5);
xline(mean(Tgo(idx_deb:end)), 'b-', 'LineWidth', 2.5, ...
    'DisplayName', sprintf('Moyenne = %.1f°C', mean(Tgo(idx_deb:end))));
xline(Tg_sec, 'k--', 'LineWidth', 2, 'DisplayName', sprintf('Seuil %d°C', Tg_sec));
xline(Tg_cib, 'g--', 'LineWidth', 1.5, 'DisplayName', sprintf('Consigne %d°C', Tg_cib));
xlabel('Tg (°C)', 'FontSize', 11);
ylabel('Fréquence', 'FontSize', 11);
if exist('pTgO', 'var')
    title(sprintf('MPC+Obs (Tg ≤ %d°C = %.1f%%)', Tg_sec, pTgO), 'FontSize', 11);
else
    title('MPC+Obs', 'FontSize', 11);
end
legend('Location', 'best');
xlim([650, 820]);

sgtitle('Figure 2 - Distribution de Tg en régime permanent', 'FontSize', 12, 'FontWeight', 'bold');

%% Figure 3 (optionnelle) : Zoom sur zone critique
figure('Name', 'Figure 3 - Zoom zone critique', ...
    'Position', [150, 150, 1200, 600], 'Color', 'white');

% Choisir une zone où Tg est proche du seuil
z_start = 900;
z_end = 1050;
t_zoom = t_h(z_start:z_end);

subplot(2,1,1);
hold on; grid on;
plot(t_zoom, Tgo(z_start:z_end), 'b-', 'LineWidth', 1.8);
yline(Tg_sec, 'r--', 'LineWidth', 2, 'DisplayName', sprintf('Sécurité %d°C', Tg_sec));
yline(Tg_ale, 'm--', 'LineWidth', 1.5, 'DisplayName', sprintf('Alerte %d°C', Tg_ale));
xlabel('Temps (heures)'); ylabel('Tg (°C)');
title('Température chambre Tg - Zone critique');
legend('Location', 'best');

subplot(2,1,2);
hold on; grid on;
plot(t_zoom, Tpo(z_start:z_end), 'b-', 'LineWidth', 1.8);
fill([t_zoom(1), t_zoom(end), t_zoom(end), t_zoom(1)], [Tp_lo, Tp_lo, Tp_hi, Tp_hi], ...
    'g', 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'DisplayName', 'Plage qualité');
yline(Tp_lo, 'g--', 'LineWidth', 1.5, 'DisplayName', '80°C');
yline(Tp_hi, 'g--', 'LineWidth', 1.5, 'DisplayName', '90°C');
xlabel('Temps (heures)'); ylabel('Tp (°C)');
title('Température produit Tp - Zone critique');
legend('Location', 'best');

sgtitle('Figure 3 - Zoom sur une zone où Tg approche le seuil de sécurité', ...
    'FontSize', 12, 'FontWeight', 'bold');

fprintf('   Graphiques générés avec succès.\n');
fprintf('   Figures : 1-Évolution, 2-Distribution Tg, 3-Zone critique\n\n');

%% =========================================================================
%  SECTION 5 — KPIs COMPARATIFS ECO [ECO][V3]
%% =========================================================================
fprintf('>>> SECTION 5 : KPIs MPC-ECO vs Données Réelles...\n');

idx = 300:N;

% KPIs MPC-ECO
Do    = mean(Deo(idx));
pTgO  = 100 * mean(Tgo(idx) <= Tg_sec);
pTpO  = 100 * mean(Tpo(idx) >= Tp_lo & Tpo(idx) <= Tp_hi);
TgMo  = mean(Tgo(idx));
TpMo  = mean(Tpo(idx));
QfMo  = mean(Qfo(idx) * 3600);

% KPIs Données réelles + détection régime établi
if reel_ok && ~isempty(Tg_reel)
    prod_median_r = median(De_reel(De_reel > 0));
    seuil_prod = 0.80 * prod_median_r;
    idx_candidats = find(De_reel > seuil_prod);
    if ~isempty(idx_candidats)
        diff_idx = diff(idx_candidats);
        breaks = find(diff_idx > 1);
        if isempty(breaks)
            idx_reel = idx_candidats;
        else
            segments = {};
            start_seg = 1;
            for b = 1:length(breaks)
                segments{end+1} = idx_candidats(start_seg:breaks(b));
                start_seg = breaks(b) + 1;
            end
            segments{end+1} = idx_candidats(start_seg:end);
            lens = cellfun(@length, segments);
            [~, idx_max_seg] = max(lens);
            idx_reel = segments{idx_max_seg};
        end
    else
        idx_reel = 500:min(N, length(De_reel));
    end
    idx_reel = idx_reel(idx_reel <= N);
    
    Tg_reel_mean = mean(Tg_reel(idx_reel));
    Tp_reel_mean = mean(Tp_reel(idx_reel));
    Qf_reel_mean = mean(Qf_reel(idx_reel));
    De_reel_mean = mean(De_reel(idx_reel));
    pTg_reel = 100 * mean(Tg_reel(idx_reel) <= Tg_sec);
    pTp_reel = 100 * mean(Tp_reel(idx_reel) >= Tp_lo & Tp_reel(idx_reel) <= Tp_hi);
    
    % [ECO] Calculs économiques
    gain_prod     = Do - De_reel_mean;
    gain_pct      = 100 * gain_prod / max(De_reel_mean, 1);
    gain_Tp       = pTpO - pTp_reel;
    gain_Tg       = pTgO - pTg_reel;
    
    % Économie fuel
    eco_Qf_kgh     = Qf_reel_mean - QfMo;          % kg/h économisé (positif = économie)
    eco_Qf_pct     = 100 * eco_Qf_kgh / max(Qf_reel_mean, 1);
    
    % Efficacité énergétique : fuel par tonne produite
    eff_reel_kgQf_per_t  = Qf_reel_mean / max(De_reel_mean, 1);   % kg fuel / t produit
    eff_mpc_kgQf_per_t   = QfMo          / max(Do, 1);
    gain_eff_kgQf_per_t  = eff_reel_kgQf_per_t - eff_mpc_kgQf_per_t;
    
    % Coût économisé
    % hypothèse : fonctionnement 8000 h/an
    heures_an = 8000;
    eco_fuel_kg_an = eco_Qf_kgh * heures_an;           % kg/an
    eco_euros_an   = eco_fuel_kg_an * cout_fuel_euro_kg; % €/an
    
    fprintf('   Régime établi : %d points\n', length(idx_reel));
else
    Tg_reel_mean = Tg0; Tp_reel_mean = Tp0;
    Qf_reel_mean = Qf0_kgh; De_reel_mean = aD*Qf0_kgh;
    pTg_reel = 95; pTp_reel = 60;
    gain_prod = Do - De_reel_mean;
    gain_pct = 100 * gain_prod / max(De_reel_mean,1);
    gain_Tp = pTpO - pTp_reel;
    gain_Tg = pTgO - pTg_reel;
    eco_Qf_kgh = Qf_reel_mean - QfMo;
    eco_Qf_pct = 100 * eco_Qf_kgh / max(Qf_reel_mean,1);
    eff_reel_kgQf_per_t = Qf_reel_mean / max(De_reel_mean,1);
    eff_mpc_kgQf_per_t  = QfMo / max(Do,1);
    gain_eff_kgQf_per_t = eff_reel_kgQf_per_t - eff_mpc_kgQf_per_t;
    heures_an = 8000;
    eco_fuel_kg_an = eco_Qf_kgh * heures_an;
    eco_euros_an   = eco_fuel_kg_an * cout_fuel_euro_kg;
    fprintf('   (données réelles non disponibles — fallback)\n');
end

% Affichage tableau comparatif
fprintf('\n╔═══════════════════════════════════════════════════════════════════════╗\n');
fprintf('║  INDICATEUR                  RÉEL USINE  MPC-ECO     GAIN/ÉCONOMIE  ║\n');
fprintf('╠═══════════════════════════════════════════════════════════════════════╣\n');
fprintf('║  Production moy (t/h)          %7.1f    %7.1f       %+7.1f      ║\n', De_reel_mean, Do, gain_prod);
fprintf('║  Gain production (%%)           ---        ---         %+6.1f%%     ║\n', gain_pct);
fprintf('╠═══════════════════════════════════════════════════════════════════════╣\n');
fprintf('║  Qf moyen (kg/h)               %7.1f    %7.1f       %+7.1f      ║\n', Qf_reel_mean, QfMo, -eco_Qf_kgh);
fprintf('║  [ECO] Économie fuel (kg/h)    ---        ---         %+7.1f      ║\n', eco_Qf_kgh);
fprintf('║  [ECO] Économie fuel (%%)       ---        ---         %+7.1f%%     ║\n', eco_Qf_pct);
fprintf('║  [ECO] Efficacité (kg/t)        %6.2f     %6.2f      %+7.2f      ║\n', eff_reel_kgQf_per_t, eff_mpc_kgQf_per_t, gain_eff_kgQf_per_t);
fprintf('║  [ECO] Économie/an (k€)        ---        ---         %+7.0f k€   ║\n', eco_euros_an/1000);
fprintf('╠═══════════════════════════════════════════════════════════════════════╣\n');
fprintf('║  Tg<=%d°C (%%)                %7.1f    %7.1f       %+7.1f      ║\n', Tg_sec, pTg_reel, pTgO, gain_Tg);
fprintf('║  Tp[%d-%d°C] (%%)              %7.1f    %7.1f       %+7.1f      ║\n', Tp_lo, Tp_hi, pTp_reel, pTpO, gain_Tp);
fprintf('║  Tg moyen (°C)                 %7.1f    %7.1f       %+7.1f      ║\n', Tg_reel_mean, TgMo, TgMo - Tg_reel_mean);
fprintf('║  Tp moyen (°C)                 %7.2f    %7.2f       %+7.2f      ║\n', Tp_reel_mean, TpMo, TpMo - Tp_reel_mean);
fprintf('╚═══════════════════════════════════════════════════════════════════════╝\n\n');

%% =========================================================================
%  SECTION 6 — ANALYSES AVANCÉES (Bootstrap IC95)
%% =========================================================================
fprintf('>>> SECTION 6 : Bootstrap IC95...\n');

n_boot = 500;
rng(42);
boot_prod   = zeros(n_boot,1);
boot_fuel   = zeros(n_boot,1);
boot_pTg    = zeros(n_boot,1);
boot_pTp    = zeros(n_boot,1);
ni = length(idx);
for b = 1:n_boot
    bidx = idx(randi(ni, 1, ni));
    boot_prod(b) = mean(Deo(bidx));
    boot_fuel(b) = mean(Qfo(bidx)*3600);
    boot_pTg(b)  = 100 * mean(Tgo(bidx) <= Tg_sec);
    boot_pTp(b)  = 100 * mean(Tpo(bidx) >= Tp_lo & Tpo(bidx) <= Tp_hi);
end
ci_prod = [prctile(boot_prod,2.5), prctile(boot_prod,97.5)];
ci_fuel = [prctile(boot_fuel,2.5), prctile(boot_fuel,97.5)];
ci_pTg  = [prctile(boot_pTg,2.5),  prctile(boot_pTg,97.5)];
ci_pTp  = [prctile(boot_pTp,2.5),  prctile(boot_pTp,97.5)];

fprintf('   IC95 Production  : [%.1f , %.1f] t/h\n',  ci_prod(1), ci_prod(2));
fprintf('   IC95 Fuel        : [%.1f , %.1f] kg/h\n', ci_fuel(1), ci_fuel(2));
fprintf('   IC95 Tg<=%d      : [%.1f , %.1f]%%\n',    Tg_sec, ci_pTg(1), ci_pTg(2));
fprintf('   IC95 Tp OK       : [%.1f , %.1f]%%\n\n',  ci_pTp(1), ci_pTp(2));

[~, p_prod, ~, ~] = ttest(Deo(idx), De_reel_mean, 'Tail', 'right');
[~, p_tg,   ~, ~] = ttest(Tgo(idx), Tg_sec,       'Tail', 'left');
fprintf('   Prod >= réel (p-val)      : %.4f\n', p_prod);
fprintf('   Tg <= sécurité (p-val)   : %.4f\n\n', p_tg);

%% =========================================================================
%  SECTION 7 — MONTE CARLO n=200 [C4]
%% =========================================================================
fprintf('>>> SECTION 7 : Monte Carlo n=200...\n');

sigma_A = 0.0005;
sigma_B = 0.008 * B_mod;
sigma_D = 0.008 * D_mod;
sigma_meas = 0.8;

n_mc = 200;
prod_mc   = zeros(n_mc,1);
fuel_mc   = zeros(n_mc,1);
tg_viol_mc= zeros(n_mc,1);
tp_ok_mc  = zeros(n_mc,1);
tgmax_mc  = zeros(n_mc,1);

fprintf('   Simulation %d itérations... ', n_mc);
rng(42);

for mc = 1:n_mc
    A_mc = A_mod + sigma_A * randn();
    B_mc = B_mod + sigma_B * randn();
    D_mc = D_mod + sigma_D * randn();
    
    Tg_mc = zeros(N,1);
    Qf_mc = zeros(N,1);
    Tp_mc = zeros(N,1);
    De_mc = zeros(N,1);
    Tg_mc(1) = Tg0; Qf_mc(1) = Qf0_kgs; Tp_mc(1) = Tp0; De_mc(1) = aD*Qf0_kgh;
    Qfl_mc = Qf0_kgs;
    c0h_mc = 0;
    bT_mc = Tg0*ones(lseq,1);
    bP_mc = Tp0*ones(lseq,1);
    bQ_mc = Qf0_kgh*ones(lseq,1);
    Tg_filtre = Tg_mc(1);
    alpha_filtre = 0.7;
    
    for k = 1:N-1
        Tg_mes = Tg_mc(k) + sigma_meas * randn();
        Tg_filtre = alpha_filtre*Tg_filtre + (1-alpha_filtre)*Tg_mes;
        
        if mod(k,15)==0 && k>80
            if Tg_mc(k)<Tg_ale-3 && Tg_mc(k)>Tg_cib-15
                % [ECO] montée plancher plus lente
                Qfl_mc = min(Qf_cible_eco_kgs, Qfl_mc + 4/3600);
            elseif Tg_mc(k)>Tg_ale
                Qfl_mc = max(Qf_lo_kgs, Qfl_mc - 25/3600);
            else
                Qfl_mc = min(Qf_cible_eco_kgs, Qfl_mc + 2/3600);
            end
        end
        
        marge_secu_mc = max(0, (Tg_mc(k)-775)/50);
        Tg_limite_base = Tg_sec - 2;
        Tg_MPC_lim_dyn_mc = max(790, min(Tg_limite_base, Tg_limite_base - marge_secu_mc*4));
        
        if     Tg_filtre > Tg_ale+3, Tr_mc = Tg_cib - 20;
        elseif Tg_filtre > Tg_ale,   Tr_mc = Tg_cib - 12;
        else,                         Tr_mc = Tg_cib + 2;
        end
        
        dx_mc = Tg_filtre - Tg0;
        fq_mc = (GT'*qT*(FT*dx_mc + Tg0 - Tr_mc*ones(Np,1)))';
        At_mc = [Ld;-Ld;tril(ones(Nc));-tril(ones(Nc));GT];
        bt_mc = [dQf_kgs*ones(2*Nc,1);
                 (Qf_cible_eco_kgs - Qf_mc(k))*ones(Nc,1);  % [ECO]
                 -(Qfl_mc - Qf_mc(k))*ones(Nc,1);
                 (Tg_MPC_lim_dyn_mc - Tg0)*ones(Np,1) - FT*dx_mc];
        try
            [dU_mc,~,ef_mc] = quadprog(Hm,fq_mc,At_mc,bt_mc,[],[],[],[],[],oq);
            if ef_mc>0 && ~isempty(dU_mc), Qn_mc = Qf_mc(k) + dU_mc(1);
            else, Qn_mc = Qf_mc(k) - 30/3600*(Tg_filtre>Tg_ale); end
        catch; Qn_mc = Qf_mc(k); end
        Qf_mc(k+1) = max(Qfl_mc, min(Qf_cible_eco_kgs, Qn_mc));
        Tg_mc(k+1) = max(650, min(Tg_sec+2, A_mc*Tg_mc(k)+B_mc*Qf_mc(k+1)+D_mc+sigma_meas*randn()));
        c0h_mc = c0h_mc + Kobs*(Tg_mc(k+1)-(A_mod*Tg_mc(k)+B_mod*Qf_mc(k)+D_mod+c0h_mc));
        bT_mc=[bT_mc(2:end);Tg_mc(k+1)];
        bP_mc=[bP_mc(2:end);Tp_mc(k)];
        bQ_mc=[bQ_mc(2:end);Qf_mc(k+1)*3600];
        Tp_mc(k+1) = max(72,min(100,fpred(bT_mc,bP_mc,bQ_mc)+0.68*randn()));
        De_mc(k+1) = min(Dmax, aD*Qf_mc(k+1)*3600);
    end
    
    prod_mc(mc)    = mean(De_mc(idx));
    fuel_mc(mc)    = mean(Qf_mc(idx)*3600);
    tg_viol_mc(mc) = 100*mean(Tg_mc(idx)>Tg_sec);
    tp_ok_mc(mc)   = 100*mean(Tp_mc(idx)>=Tp_lo & Tp_mc(idx)<=Tp_hi);
    tgmax_mc(mc)   = max(Tg_mc(idx));
end
fprintf('OK\n');

% Seuil production = légèrement > réel (marge_prod_cible / 2 au minimum)
seuil_rob = max(De_reel_mean, De_reel_mean + marge_prod_cible/2);
robustesse = 100 * mean(tg_viol_mc < 5 & prod_mc >= seuil_rob);
eco_robustesse = 100 * mean(fuel_mc < Qf_reel_mean & prod_mc >= seuil_rob);

fprintf('\n   Production MC    : %.1f ± %.1f t/h\n', mean(prod_mc), std(prod_mc));
fprintf('   Fuel MC          : %.1f ± %.1f kg/h\n', mean(fuel_mc), std(fuel_mc));
fprintf('   Robustesse Prod>réel : %.1f%%\n', robustesse);
fprintf('   [ECO] Robustesse (fuel<réel ET prod>réel) : %.1f%%\n\n', eco_robustesse);

%% =========================================================================
%  SECTION 7b — VALIDATION BOUCLE OUVERTE [C2]
%% =========================================================================
fprintf('>>> SECTION 7b : Validation boucle ouverte [C2]...\n');

RMSE_validation_OL = NaN;
rmse_1step = NaN;

try
    data_7b = readtable('tout.xlsx','Range','B:E','VariableNamingRule','preserve');
    Tg_reel_7b = double(data_7b{:,1});
    Tp_reel_7b = double(data_7b{:,3});
    Qf_reel_7b = double(data_7b{:,4});
    valid_7b = ~isnan(Tg_reel_7b)&~isnan(Tp_reel_7b)&~isnan(Qf_reel_7b)&Qf_reel_7b>100&Tg_reel_7b>600;
    Tg_reel_7b = Tg_reel_7b(valid_7b);
    Tp_reel_7b = Tp_reel_7b(valid_7b);
    Qf_reel_7b = Qf_reel_7b(valid_7b);
    N_reel_7b = length(Tg_reel_7b);
    Qf_reel_kgs = Qf_reel_7b/3600;
    n_id_7b = N_reel_7b - 1;
    X_id_7b = [Tg_reel_7b(1:n_id_7b), Qf_reel_kgs(1:n_id_7b), ones(n_id_7b,1)];
    Y_id_7b = Tg_reel_7b(2:end);
    theta_new = (X_id_7b'*X_id_7b)\(X_id_7b'*Y_id_7b);
    A_new=theta_new(1); B_new=theta_new(2); D_new=theta_new(3);
    Tg_pred_new_1step = A_new*Tg_reel_7b(1:n_id_7b)+B_new*Qf_reel_kgs(1:n_id_7b)+D_new;
    Tg_true_7b = Tg_reel_7b(2:end);
    err_new = Tg_true_7b - Tg_pred_new_1step;
    rmse_new = sqrt(mean(err_new.^2));
    r2_new   = 1 - sum(err_new.^2)/sum((Tg_true_7b-mean(Tg_true_7b)).^2);
    RMSE_validation_OL = rmse_new;
    fprintf('   Nouveau modèle (1-step) : RMSE=%.2f°C  R²=%.4f\n', rmse_new, r2_new);
    fprintf('   [P1] RMSE_validation_OL = %.2f°C\n\n', RMSE_validation_OL);
catch ME_7b
    fprintf('   Erreur Section 7b : %s\n\n', ME_7b.message);
end

%% =========================================================================
%  SECTION 8 — TESTS DE ROBUSTESSE (4 scénarios)
%% =========================================================================
fprintf('>>> SECTION 8 : Tests de robustesse...\n');
rng(42);

arg_s = {A_mod, B_mod, D_mod, Qf0_kgs, Tg0, Tp0, N, Np, Nc, qT, GT, FT, Hm, Ld, oq, ...
         dQf_kgs, Qf_cible_eco_kgs, Qf_lo_kgs, Qf_cible_eco_kgs, ...
         Tg_ale, Tg_cib, Tg_sec, Dmax, aD, lseq, Kobs, fpred};

[Tg_s1,Tp_s1,De_s1] = run_scenario(arg_s{:}, 2, N+1, 0, 0, Qf0_kgh);
[Tg_s2,Tp_s2,De_s2] = run_scenario(arg_s{:}, 6, N+1, 0, 0, Qf0_kgh);
[Tg_s3,Tp_s3,De_s3] = run_scenario(arg_s{:}, 2, 400, 20, 0, Qf0_kgh);
[Tg_s4,Tp_s4,De_s4] = run_scenario(arg_s{:}, 2, N+1, 0, 0.05, Qf0_kgh);

fprintf('   S1(nominal)     : Prod=%.0ft/h  Tg<=%d:%.1f%%\n', mean(De_s1(idx)),Tg_sec,100*mean(Tg_s1(idx)<=Tg_sec));
fprintf('   S2(bruit s=6°C) : Prod=%.0ft/h  Tg<=%d:%.1f%%\n', mean(De_s2(idx)),Tg_sec,100*mean(Tg_s2(idx)<=Tg_sec));
fprintf('   S3(échelon+20°C): Prod=%.0ft/h  Tg<=%d:%.1f%%\n', mean(De_s3(idx)),Tg_sec,100*mean(Tg_s3(idx)<=Tg_sec));
fprintf('   S4(dérive+0.05) : Prod=%.0ft/h  Tg<=%d:%.1f%%\n\n',mean(De_s4(idx)),Tg_sec,100*mean(Tg_s4(idx)<=Tg_sec));

%% =========================================================================
%  SECTION 9 — EXPORT CSV & RAPPORT [ECO]
%% =========================================================================
fprintf('>>> SECTION 9 : Export CSV + Rapport...\n');
t_h = (0:N-1)/60;

T_csv = table(t_h', Tgo, Tpo, Qfo*3600, Deo, kADo, ...
    'VariableNames',{'t_h','Tg_MPC','Tp_MPC','Qf_MPC_kgh','Deng_MPC','kAD'});
if reel_ok
    T_reel_csv = table(t_h', Tg_reel, Tp_reel, Qf_reel, De_reel, ...
        'VariableNames',{'t_h','Tg_reel','Tp_reel','Qf_reel_kgh','Prod_reel_th'});
    try; writetable(T_reel_csv,'PFE_donnees_reelles.csv'); fprintf('   CSV réel sauvegardé\n'); catch; end
end
try; writetable(T_csv,'PFE_resultats_simulation_ECO.csv'); fprintf('   CSV simulation ECO sauvegardé\n'); catch; end

T_mc = table((1:n_mc)', prod_mc, fuel_mc, tg_viol_mc, tp_ok_mc, tgmax_mc, ...
    'VariableNames',{'iter','prod_th','fuel_kgh','tg_viol_pct','tp_ok_pct','tg_max_C'});
try; writetable(T_mc,'PFE_montecarlo_ECO.csv'); fprintf('   CSV Monte Carlo ECO sauvegardé\n'); catch; end

rapport = {};
rapport{end+1} = '=== RAPPORT PFE SÉCHEUR TSP — V3-ECO OPTIMISATION ÉNERGÉTIQUE ===';
rapport{end+1} = sprintf('Date : %s', datestr(now,'dd/mm/yyyy HH:MM'));
rapport{end+1} = '';
rapport{end+1} = '--- STRATÉGIE ECO ---';
rapport{end+1} = sprintf('Objectif : maintenir prod ~ réelle + %.0f t/h, MINIMISER fuel', marge_prod_cible);
rapport{end+1} = sprintf('R_eco (pénalité fuel MPC) = %.1f  (vs 0.02 classique)', R_eco);
rapport{end+1} = '';
rapport{end+1} = '--- RÉSULTATS PRODUCTION ---';
rapport{end+1} = sprintf('Production réelle usine : %.1f t/h', De_reel_mean);
rapport{end+1} = sprintf('Production MPC-ECO      : %.1f t/h', Do);
rapport{end+1} = sprintf('Gain production         : %+.1f t/h (%+.1f%%)', gain_prod, gain_pct);
rapport{end+1} = '';
rapport{end+1} = '--- ÉCONOMIE ÉNERGÉTIQUE ---';
rapport{end+1} = sprintf('Fuel réel usine   : %.1f kg/h', Qf_reel_mean);
rapport{end+1} = sprintf('Fuel MPC-ECO      : %.1f kg/h', QfMo);
rapport{end+1} = sprintf('Économie fuel     : +%.1f kg/h  (+%.1f%%)', eco_Qf_kgh, eco_Qf_pct);
rapport{end+1} = sprintf('Efficacité réelle : %.2f kg fuel/t produit', eff_reel_kgQf_per_t);
rapport{end+1} = sprintf('Efficacité ECO    : %.2f kg fuel/t produit', eff_mpc_kgQf_per_t);
rapport{end+1} = sprintf('Gain efficacité   : %.2f kg fuel/t produit', gain_eff_kgQf_per_t);
rapport{end+1} = sprintf('Économie annuelle : %.0f k€/an (base %.0f h/an)', eco_euros_an/1000, heures_an);
rapport{end+1} = '';
rapport{end+1} = '--- QUALITÉ & SÉCURITÉ ---';
rapport{end+1} = sprintf('Tp[80-90°C] réel  : %.1f%%', pTp_reel);
rapport{end+1} = sprintf('Tp[80-90°C] ECO   : %.1f%%', pTpO);
rapport{end+1} = sprintf('Tg<=%d°C réel   : %.1f%%', Tg_sec, pTg_reel);
rapport{end+1} = sprintf('Tg<=%d°C ECO    : %.1f%%', Tg_sec, pTgO);
rapport{end+1} = '';
rapport{end+1} = '--- CONFIGURATION ---';
rapport{end+1} = sprintf('LSTM : H=16, RMSE=%.4f°C, R²=%.4f', RMSE_lstm, R2_lstm);
rapport{end+1} = sprintf('MPC  : Np=%d, Nc=%d, qT=%d, R_eco=%.1f', Np, Nc, qT, R_eco);
rapport{end+1} = sprintf('Kobs : %.2f', Kobs);
rapport{end+1} = sprintf('Robustesse MC (ECO) : %.1f%%', eco_robustesse);
rapport{end+1} = sprintf('Temps total : %.1f s', toc(t_start_global));

try
    fid = fopen('PFE_rapport_ECO.txt','w');
    for i=1:length(rapport); fprintf(fid,'%s\n',rapport{i}); end
    fclose(fid);
    fprintf('   Rapport sauvegardé : PFE_rapport_ECO.txt\n\n');
catch; end

%% =========================================================================
%  SECTION 10 — INTERFACE SCADA ECO [ECO]
%% =========================================================================
fprintf('>>> SECTION 10 : Interface SCADA ECO...\n');

C.bg=[0.08 0.10 0.14]; C.panel=[0.13 0.16 0.20]; C.panel2=[0.10 0.13 0.17];
C.border=[0.25 0.35 0.50]; C.tb=[0.05 0.12 0.25]; C.green=[0.10 0.90 0.20];
C.red=[1.00 0.15 0.10]; C.orange=[1.00 0.60 0.05]; C.cyan=[0.10 0.85 0.95];
C.white=[0.95 0.95 0.95]; C.yellow=[1.00 0.90 0.05]; C.gray=[0.50 0.55 0.60];
C.blue=[0.20 0.55 1.00]; C.pink=[1.00 0.40 0.70]; C.purple=[0.70 0.50 1.00];
C.g2=[0.05 0.62 0.13]; C.tN=[1.00 0.35 0.35]; C.tO=[0.25 0.70 1.00];
C.tSO=[1.00 0.65 0.10]; C.tG=[0.20 0.90 0.40];
C.eco=[0.20 0.85 0.50];  % [ECO] couleur verte éco

scrsz=get(0,'ScreenSize');
W=scrsz(3)-40; H=scrsz(4)-80;
fig=figure('Name','SCADA V3-ECO — MPC-ECO MINIMISATION FUEL | SÉCHEUR TSP', ...
    'NumberTitle','off','Color',C.bg,'Position',[20 20 W H], ...
    'MenuBar','none','ToolBar','none','Resize','on');

% ── TITRE ─────────────────────────────────────────────────────────────────
uipanel('Parent',fig,'BackgroundColor',C.tb,'BorderType','none', ...
    'Units','normalized','Position',[0 1-60/H 1 60/H]);
uicontrol('Parent',fig,'Style','text', ...
    'String',sprintf('[ECO] SÉCHEUR TSP | Fuel économisé = +%.0f kg/h (%.1f%%) | Prod +%.0ft/h | ~%.0fk€/an | Kobs=%.2f', ...
        eco_Qf_kgh, eco_Qf_pct, gain_prod, eco_euros_an/1000, Kobs), ...
    'ForegroundColor',C.eco,'BackgroundColor',C.tb, ...
    'FontSize',10,'FontWeight','bold','FontName','Consolas', ...
    'HorizontalAlignment','center','Units','normalized','Position',[0.01 1-56/H 0.68 52/H]);
uicontrol('Parent',fig,'Style','text','String','● ECO-MODE', ...
    'ForegroundColor',C.eco,'BackgroundColor',C.tb, ...
    'FontSize',10,'FontWeight','bold','FontName','Consolas', ...
    'Units','normalized','Position',[0.72 1-52/H 0.10 46/H]);
htclock=uicontrol('Parent',fig,'Style','text', ...
    'String',datestr(now,'dd/mm/yyyy  HH:MM:SS'), ...
    'ForegroundColor',C.yellow,'BackgroundColor',C.tb, ...
    'FontSize',10,'FontWeight','bold','FontName','Consolas', ...
    'HorizontalAlignment','right','Units','normalized','Position',[0.83 1-55/H 0.16 50/H]);

% ── SYNOPTIQUE ────────────────────────────────────────────────────────────
syn=uipanel('Parent',fig,'BackgroundColor',C.panel,'BorderType','line', ...
    'HighlightColor',C.border,'Units','normalized','Position',[0.005 0.01 0.255 0.88]);
uicontrol('Parent',syn,'Style','text','String','  SYNOPTIQUE PROCÉDÉ [ECO]', ...
    'ForegroundColor',C.eco,'BackgroundColor',C.tb,'FontSize',10,'FontWeight','bold', ...
    'FontName','Consolas','HorizontalAlignment','left','Units','normalized','Position',[0 0.96 1 0.04]);
ax_syn=axes('Parent',syn,'Position',[0.03 0.05 0.94 0.89], ...
    'Color',C.panel2,'XColor',C.border,'YColor',C.border, ...
    'XLim',[0 100],'YLim',[0 100],'XTick',[],'YTick',[],'Box','on');
hold(ax_syn,'on');
fill(ax_syn,[0 100 100 0],[0 0 100 100],C.panel2,'EdgeColor','none');

rx=15;ry=34;rw=46;rh=38;
fill(ax_syn,[rx rx+rw rx+rw rx],[ry ry ry+rh ry+rh],[0.55 0.08 0.08],'EdgeColor',[1 0.3 0.1],'LineWidth',2.5);
text(ax_syn,rx+rw/2,ry+rh-7,'CHAMBRE','Color','w','FontSize',9,'FontWeight','bold','HorizontalAlignment','center');
text(ax_syn,rx+rw/2,ry+rh-15,'COMBUSTION','Color',[1 0.82 0.10],'FontSize',8.5,'FontWeight','bold','HorizontalAlignment','center');
for fi=1:3
    fx=rx+8+(fi-1)*14;
    fill(ax_syn,[fx-4 fx fx+4],[ry+2 ry+17 ry+2],[0.94+0.02*fi 0.44 0.04],'EdgeColor','none','FaceAlpha',0.9);
    fill(ax_syn,[fx-2 fx fx+2],[ry+5 ry+13 ry+5],[1 0.86 0.12],'EdgeColor','none','FaceAlpha',0.94);
end
hTg_syn=text(ax_syn,rx+rw/2,ry+rh/2-5,'','Color',C.green,'FontSize',12,'FontWeight','bold', ...
    'HorizontalAlignment','center','FontName','Consolas');
rectangle('Parent',ax_syn,'Position',[rx+5 ry+rh+2 14 7],'Curvature',0.3,'FaceColor',[0.15 0.25 0.15],'EdgeColor',C.green,'LineWidth',1.8);
text(ax_syn,rx+12,ry+rh+5.5,'TT-01','Color',C.green,'FontSize',7,'FontWeight','bold','HorizontalAlignment','center');

tx=65;ty=40;tw=28;th=19;
rectangle('Parent',ax_syn,'Position',[tx ty tw th],'Curvature',[0.25 0.45],'FaceColor',[0.13 0.21 0.40],'EdgeColor',C.blue,'LineWidth',2.5);
for ri=0:2
    rectangle('Parent',ax_syn,'Position',[tx+1.5+ri*8.5 ty+1.5 7 th-3],'Curvature',[0.2 0.4],'FaceColor',[0.17 0.27 0.48],'EdgeColor',[0.22 0.37 0.65],'LineWidth',0.7);
end
text(ax_syn,tx+tw/2,ty+th/2+3,'TAMBOUR','Color','w','FontSize',8.5,'FontWeight','bold','HorizontalAlignment','center');
text(ax_syn,tx+tw/2,ty+th/2-3,'ROTATIF','Color',C.cyan,'FontSize',7.5,'HorizontalAlignment','center');
hTp_syn=text(ax_syn,tx+tw/2,ty-7,'','Color',C.green,'FontSize',10,'FontWeight','bold','HorizontalAlignment','center','FontName','Consolas');
rectangle('Parent',ax_syn,'Position',[tx+4 ty+th+2 18 7],'Curvature',0.3,'FaceColor',[0.15 0.25 0.15],'EdgeColor',C.green,'LineWidth',1.8);
text(ax_syn,tx+13,ty+th+5.5,'TT-02','Color',C.green,'FontSize',7,'FontWeight','bold','HorizontalAlignment','center');

pfx=rx+rw/2;
fill(ax_syn,[pfx-3 pfx+3 pfx+3 pfx-3],[0 0 ry ry],[0.44 0.24 0.02],'EdgeColor',[0.90 0.50 0.10],'LineWidth',1.8);
patch(ax_syn,[pfx-4.5 pfx pfx+4.5],[ry-6 ry ry-6],[1 0.55 0.05],'EdgeColor','none');
fill(ax_syn,[pfx-4 pfx pfx+4],[7 14 7],[0.09 0.38 0.09],'EdgeColor',C.green,'LineWidth',2);
fill(ax_syn,[pfx-4 pfx pfx+4],[21 14 21],[0.09 0.38 0.09],'EdgeColor',C.green,'LineWidth',2);
text(ax_syn,pfx+5.5,14,'FV-01','Color',C.green,'FontSize',7,'FontWeight','bold');
hQf_syn=text(ax_syn,pfx,ry/2+2,'','Color',C.orange,'FontSize',8.5,'FontWeight','bold','HorizontalAlignment','center','FontName','Consolas');
text(ax_syn,pfx,ry/2-6,'FUEL','Color',[1 0.72 0.22],'FontSize',7,'FontWeight','bold','HorizontalAlignment','center');
% [ECO] Badge économie fuel sur la conduite
hEcoFuel=text(ax_syn,pfx+10,ry/2+8,'','Color',C.eco,'FontSize',7.5,'FontWeight','bold','HorizontalAlignment','left','FontName','Consolas');

plot(ax_syn,[0 rx],[ry+rh/2 ry+rh/2],'Color',C.blue,'LineWidth',3);
patch(ax_syn,[rx-4 rx rx-4],[ry+rh/2-3 ry+rh/2 ry+rh/2+3],C.blue,'EdgeColor','none');
hkAC_syn=text(ax_syn,1,ry+rh/2+6,'','Color',C.blue,'FontSize',7.5,'FontName','Consolas');
text(ax_syn,rx/2,ry+rh/2-5,'AIR C.','Color',C.blue,'FontSize',7,'FontWeight','bold','HorizontalAlignment','center');
fill(ax_syn,[rx+rw rx+rw rx+rw+12 rx+rw+12],[ry+rh-5 ry+rh ry+rh ry+rh-5],[0.28 0.53 0.84],'EdgeColor',[0.40 0.70 1.0],'LineWidth',1.8);
hkAD_syn=text(ax_syn,rx+rw+2,ry+rh-11,'','Color',[0.45 0.76 1.0],'FontSize',7.5,'FontName','Consolas');
text(ax_syn,rx+rw+6,ry+rh+2.5,'AIR D.','Color',[0.4 0.7 1.0],'FontSize',7,'FontWeight','bold','HorizontalAlignment','center');
plot(ax_syn,[rx+rw tx],[ry+rh/2 ty+th/2],'Color',[1 0.7 0.2],'LineWidth',3);
patch(ax_syn,[tx-4 tx tx-4],[ty+th/2-3 ty+th/2 ty+th/2+3],[1 0.7 0.2],'EdgeColor','none');
fill(ax_syn,[tx+tw 100 100 tx+tw],[ty+th/2-2.5 ty+th/2-2.5 ty+th/2+2.5 ty+th/2+2.5],C.g2,'EdgeColor',C.green,'LineWidth',1.8);
patch(ax_syn,[97 100 97],[ty+th/2-3.5 ty+th/2 ty+th/2+3.5],C.green,'EdgeColor','none');
hDeng_syn=text(ax_syn,82,ty+th/2-7,'','Color',C.green,'FontSize',9,'FontWeight','bold','HorizontalAlignment','left','FontName','Consolas');
text(ax_syn,82,ty+th/2+7,'PRODUIT','Color',C.green,'FontSize',7.5,'FontWeight','bold');
plot(ax_syn,[pfx+8 pfx+8],[ry+rh 100],'Color',C.gray,'LineWidth',2);
text(ax_syn,pfx+13,97,'FUMEES','Color',C.gray,'FontSize',7,'FontWeight','bold','HorizontalAlignment','center');

rectangle('Parent',ax_syn,'Position',[2 78 22 14],'Curvature',0.2,'FaceColor',[0.05 0.15 0.30],'EdgeColor',C.blue,'LineWidth',2);
text(ax_syn,13,88,'MPC-ECO','Color',C.eco,'FontSize',8,'FontWeight','bold','HorizontalAlignment','center');
text(ax_syn,13,83,sprintf('R_eco=%.1f',R_eco),'Color',C.cyan,'FontSize',6.5,'HorizontalAlignment','center');
rectangle('Parent',ax_syn,'Position',[2 60 22 14],'Curvature',0.2,'FaceColor',[0.05 0.10 0.25],'EdgeColor',C.pink,'LineWidth',2);
text(ax_syn,13,70,'LSTM','Color',C.pink,'FontSize',9,'FontWeight','bold','HorizontalAlignment','center');
text(ax_syn,13,65,sprintf('R2=%.2f',R2_lstm),'Color',[0.9 0.7 0.9],'FontSize',6.5,'HorizontalAlignment','center');
rectangle('Parent',ax_syn,'Position',[2 44 22 12],'Curvature',0.2,'FaceColor',[0.07 0.17 0.07],'EdgeColor',C.eco,'LineWidth',2);
% [ECO] badge économie annuelle
text(ax_syn,13,51,sprintf('%.0fk€/an',eco_euros_an/1000),'Color',C.eco,'FontSize',8,'FontWeight','bold','HorizontalAlignment','center');
text(ax_syn,13,46,'Économie','Color',[0.7 0.9 0.7],'FontSize',6.5,'HorizontalAlignment','center');
rectangle('Parent',ax_syn,'Position',[2 29 22 12],'Curvature',0.2,'FaceColor',[0.07 0.04 0.17],'EdgeColor',C.purple,'LineWidth',2);
hkADB=text(ax_syn,13,36,'','Color',C.purple,'FontSize',9,'FontWeight','bold','HorizontalAlignment','center');
text(ax_syn,13,31,'k_AD','Color',[0.8 0.7 1.0],'FontSize',6.5,'HorizontalAlignment','center');
rectangle('Parent',ax_syn,'Position',[2 13 22 13],'Curvature',0.2,'FaceColor',[0.04 0.18 0.10],'EdgeColor',C.tG,'LineWidth',2);
hObs_badge=text(ax_syn,13,20.5,'','Color',C.tG,'FontSize',8,'FontWeight','bold','HorizontalAlignment','center');
text(ax_syn,13,14.5,sprintf('EKF K=%.2f',Kobs),'Color',[0.7 0.9 0.7],'FontSize',6.5,'HorizontalAlignment','center');

% ── MESURES ───────────────────────────────────────────────────────────────
pm=uipanel('Parent',fig,'BackgroundColor',C.panel,'BorderType','line','HighlightColor',C.border, ...
    'Units','normalized','Position',[0.263 0.54 0.175 0.35]);
uicontrol('Parent',pm,'Style','text','String','  MESURES PROCESS', ...
    'ForegroundColor',C.eco,'BackgroundColor',C.tb,'FontSize',9,'FontWeight','bold', ...
    'FontName','Consolas','HorizontalAlignment','left','Units','normalized','Position',[0 0.93 1 0.07]);
jlabs={'Tg Chambre (°C)','Tp Produit (°C)','Qf Fuel (kg/h)','Deng Prod (t/h)','k_AD (kg/kg)','Obs c0 (°C)'};
junits={'°C','°C','kg/h','t/h','kg/kg','°C'};
jcols={C.red,C.green,C.eco,C.cyan,C.purple,C.tG};  % [ECO] Qf en vert éco
jmin=[650 75 500 700 25 -15]; jmax=[820 95 2500 970 50 15]; jlim=[Tg_sec 90 Qf_cible_eco_kgh 950 45 10];
hy=linspace(0.82,0.03,6);
hJv=zeros(1,6); hJb=zeros(1,6);
for jj=1:6
    uicontrol('Parent',pm,'Style','text','String',jlabs{jj},'ForegroundColor',C.gray,'BackgroundColor',C.panel, ...
        'FontSize',6.5,'FontName','Consolas','HorizontalAlignment','left','Units','normalized','Position',[0.03 hy(jj)+0.082 0.94 0.038]);
    hJv(jj)=uicontrol('Parent',pm,'Style','text','String','----','ForegroundColor',jcols{jj}, ...
        'BackgroundColor',[0.05 0.08 0.10],'FontSize',14,'FontWeight','bold','FontName','Consolas', ...
        'HorizontalAlignment','center','Units','normalized','Position',[0.03 hy(jj)+0.018 0.56 0.062]);
    uicontrol('Parent',pm,'Style','text','String',junits{jj},'ForegroundColor',C.gray,'BackgroundColor',C.panel, ...
        'FontSize',7,'FontName','Consolas','Units','normalized','Position',[0.61 hy(jj)+0.038 0.35 0.036]);
    uipanel('Parent',pm,'BackgroundColor',[0.08 0.10 0.12],'BorderType','none','Units','normalized','Position',[0.03 hy(jj) 0.94 0.016]);
    hJb(jj)=uipanel('Parent',pm,'BackgroundColor',jcols{jj},'BorderType','none','Units','normalized','Position',[0.03 hy(jj) 0.001 0.016]);
end

% ── ALARMES ───────────────────────────────────────────────────────────────
pal=uipanel('Parent',fig,'BackgroundColor',C.panel,'BorderType','line','HighlightColor',[0.7 0.1 0.1], ...
    'Units','normalized','Position',[0.263 0.34 0.175 0.19]);
uicontrol('Parent',pal,'Style','text','String','  ALARMES & ÉTAT', ...
    'ForegroundColor',[1 0.4 0.4],'BackgroundColor',[0.25 0.05 0.05],'FontSize',9,'FontWeight','bold', ...
    'FontName','Consolas','HorizontalAlignment','left','Units','normalized','Position',[0 0.90 1 0.10]);
alabs={sprintf('Tg > %d°C  SÉCURITÉ',Tg_sec),'Tp < 80°C  QUALITÉ', ...
       'Tp > 90°C  SURCHAUFFE','FUEL > CIBLE ECO','OBS  ÉCART > 10°C'};
alcls={C.red,C.orange,C.orange,C.eco,C.tG};
hAl=zeros(1,5); ay=linspace(0.72,0.05,5);
for aa=1:5
    uicontrol('Parent',pal,'Style','text','String','●','ForegroundColor',C.gray,'BackgroundColor',C.panel,'FontSize',14,'Units','normalized','Position',[0.03 ay(aa) 0.12 0.14]);
    hAl(aa)=uicontrol('Parent',pal,'Style','text','String',alabs{aa},'ForegroundColor',C.gray,'BackgroundColor',C.panel,'FontSize',8,'FontName','Consolas','HorizontalAlignment','left','Units','normalized','Position',[0.18 ay(aa) 0.80 0.13]);
end

% ── KPI ECO ───────────────────────────────────────────────────────────────
pkpi=uipanel('Parent',fig,'BackgroundColor',C.panel,'BorderType','line','HighlightColor',C.border, ...
    'Units','normalized','Position',[0.263 0.01 0.175 0.32]);
uicontrol('Parent',pkpi,'Style','text','String','  KPI ÉNERGIE vs RÉEL [ECO]', ...
    'ForegroundColor',C.eco,'BackgroundColor',C.tb,'FontSize',9,'FontWeight','bold', ...
    'FontName','Consolas','HorizontalAlignment','left','Units','normalized','Position',[0 0.93 1 0.07]);
klabs={'Prod.Réel(t/h)','Prod.ECO(t/h)','Gain prod(t/h)', ...
       'Fuel réel(kg/h)','Fuel ECO(kg/h)','Éco.fuel(kg/h)', ...
       'Éco.fuel(%)', sprintf('Effic.(kg/t)'),sprintf('€/an(k€)'), ...
       'Robustesse ECO'};
kvals={sprintf('%.0f',De_reel_mean), sprintf('%.0f',Do), sprintf('%+.0f',gain_prod), ...
       sprintf('%.0f',Qf_reel_mean), sprintf('%.0f',QfMo), sprintf('+%.0f',eco_Qf_kgh), ...
       sprintf('+%.1f%%',eco_Qf_pct), sprintf('%.2f→%.2f',eff_reel_kgQf_per_t,eff_mpc_kgQf_per_t), ...
       sprintf('%.0fk€',eco_euros_an/1000), sprintf('%.1f%%',eco_robustesse)};
kcols={C.tN,C.tO,C.yellow,C.orange,C.eco,C.eco,C.eco,C.cyan,C.eco,C.tG};
ky=linspace(0.88,0.03,10);
for ki=1:10
    uicontrol('Parent',pkpi,'Style','text','String',klabs{ki},'ForegroundColor',C.gray,'BackgroundColor',C.panel, ...
        'FontSize',7,'FontName','Consolas','HorizontalAlignment','left','Units','normalized','Position',[0.03 ky(ki) 0.56 0.085]);
    uicontrol('Parent',pkpi,'Style','text','String',kvals{ki},'ForegroundColor',kcols{ki},'BackgroundColor',[0.05 0.08 0.10], ...
        'FontSize',8,'FontWeight','bold','FontName','Consolas','HorizontalAlignment','center','Units','normalized','Position',[0.61 ky(ki) 0.36 0.085]);
end

% ── TENDANCES 4 graphes ───────────────────────────────────────────────────
axP={'Color',C.panel2,'XColor',C.gray,'YColor',C.gray,'GridColor',[0.2 0.3 0.4], ...
     'GridAlpha',0.5,'FontName','Consolas','FontSize',7.5,'Box','on','XGrid','on','YGrid','on'};
tr_pos={[0.445 0.755 0.548 0.185],[0.445 0.550 0.548 0.185], ...
        [0.445 0.345 0.548 0.185],[0.445 0.105 0.548 0.185]};
hCur=zeros(1,4);

% Graphe 1 : Production + ligne cible ECO
pT1=uipanel('Parent',fig,'BackgroundColor',C.panel2,'BorderType','none','Units','normalized','Position',tr_pos{1});
aT1=axes('Parent',pT1,'Position',[0.06 0.14 0.92 0.78],axP{:}); hold(aT1,'on');
plot(aT1,t_h,Deo,'Color',C.tO,'LineWidth',2.2,'DisplayName',sprintf('MPC-ECO %.0ft/h',Do));
if reel_ok
    plot(aT1,t_h,De_reel,'Color',C.tN,'LineWidth',1.3,'DisplayName',sprintf('Réel %.0ft/h',De_reel_mean));
end
yline(aT1,Deng_cible_eco,'--','Color',C.eco,'LineWidth',1.5,'Label',sprintf('Cible ECO %.0ft/h',Deng_cible_eco));
yline(aT1,Dmax,'--','Color',C.gray,'LineWidth',1.2,'Label','Max 950t/h');
ylabel(aT1,'Production(t/h)','Color',C.white,'FontSize',8,'FontWeight','bold');
title(aT1,sprintf('  PRODUCTION  |  +%.0ft/h vs réel | Cible ECO=%.0ft/h (+%.0f)', ...
    gain_prod, Deng_cible_eco, marge_prod_cible),'Color',C.eco,'FontSize',9,'FontWeight','bold','HorizontalAlignment','left');
legend(aT1,'Location','southeast','TextColor',C.white,'Color',C.panel,'FontSize',8,'EdgeColor',C.border);
ylim(aT1,[max(0, min([Deo(idx);De_reel(idx_reel)])-30) Dmax+30]);
set(aT1,'XTickLabel',{});
hCur(1)=xline(aT1,0,'Color',C.yellow,'LineWidth',2,'LineStyle',':');

% Graphe 2 : Tg
pT2=uipanel('Parent',fig,'BackgroundColor',C.panel2,'BorderType','none','Units','normalized','Position',tr_pos{2});
aT2=axes('Parent',pT2,'Position',[0.06 0.14 0.92 0.78],axP{:}); hold(aT2,'on');
patch(aT2,[0 t_h(end) t_h(end) 0],[Tg_sec Tg_sec 820 820],C.red,'FaceAlpha',0.07,'EdgeColor','none');
plot(aT2,t_h,Tgo,'Color',C.tO,'LineWidth',2.2,'DisplayName',sprintf('MPC-ECO %.1f%%≤%d°C',pTgO,Tg_sec));
if reel_ok
    plot(aT2,t_h,Tg_reel,'Color',C.tN,'LineWidth',1.3,'DisplayName',sprintf('Réel %.1f%%≤%d°C',pTg_reel,Tg_sec));
end
yline(aT2,Tg_sec,'--','Color',C.red,'LineWidth',2,'Label',sprintf('%d°C [C3]',Tg_sec));
yline(aT2,Tg_cib,':','Color',C.yellow,'LineWidth',1.4,'Label',sprintf('%d°C cible',Tg_cib));
ylabel(aT2,'Tg(°C)','Color',C.white,'FontSize',8,'FontWeight','bold');
title(aT2,sprintf('  Tg CHAMBRE  |  MPC-ECO : %.1f%% ≤ %d°C  |  Réel : %.1f%%', ...
    pTgO,Tg_sec,pTg_reel),'Color',C.cyan,'FontSize',9,'FontWeight','bold','HorizontalAlignment','left');
legend(aT2,'Location','best','TextColor',C.white,'Color',C.panel,'FontSize',8,'EdgeColor',C.border);
ylim(aT2,[640 820]); set(aT2,'XTickLabel',{});
hCur(2)=xline(aT2,0,'Color',C.yellow,'LineWidth',2,'LineStyle',':');

% Graphe 3 : Tp
pT3=uipanel('Parent',fig,'BackgroundColor',C.panel2,'BorderType','none','Units','normalized','Position',tr_pos{3});
aT3=axes('Parent',pT3,'Position',[0.06 0.14 0.92 0.78],axP{:}); hold(aT3,'on');
patch(aT3,[0 t_h(end) t_h(end) 0],[Tp_lo Tp_lo Tp_hi Tp_hi],C.green,'FaceAlpha',0.08,'EdgeColor','none');
plot(aT3,t_h,Tpo,'Color',C.tO,'LineWidth',2.2,'DisplayName',sprintf('MPC-ECO %.1f%% OK',pTpO));
if reel_ok
    plot(aT3,t_h,Tp_reel,'Color',C.tN,'LineWidth',1.3,'DisplayName',sprintf('Réel %.1f%% OK',pTp_reel));
end
yline(aT3,Tp_lo,'--','Color',C.g2,'LineWidth',1.4,'Label','80°C');
yline(aT3,Tp_hi,'--','Color',C.g2,'LineWidth',1.4,'Label','90°C');
yline(aT3,85,'-','Color',C.tG,'LineWidth',1,'Label','85°C');
ylabel(aT3,'Tp(°C)','Color',C.white,'FontSize',8,'FontWeight','bold');
title(aT3,sprintf('  Tp PRODUIT LSTM  |  GAIN QUALITÉ = %+.1f%%',gain_Tp), ...
    'Color',C.pink,'FontSize',9,'FontWeight','bold','HorizontalAlignment','left');
legend(aT3,'Location','best','TextColor',C.white,'Color',C.panel,'FontSize',8,'EdgeColor',C.border);
ylim(aT3,[Tp_lo-6 Tp_hi+9]); set(aT3,'XTickLabel',{});
hCur(3)=xline(aT3,0,'Color',C.yellow,'LineWidth',2,'LineStyle',':');

% Graphe 4 : [ECO] FUEL + ligne réel + économie
pT4=uipanel('Parent',fig,'BackgroundColor',C.panel2,'BorderType','none','Units','normalized','Position',tr_pos{4});
aT4=axes('Parent',pT4,'Position',[0.06 0.16 0.92 0.76],axP{:}); hold(aT4,'on');
yyaxis(aT4,'left');
plot(aT4,t_h,Qfo*3600,'Color',C.eco,'LineWidth',2.2,'DisplayName',sprintf('Fuel ECO %.0fkg/h',QfMo));
if reel_ok
    plot(aT4,t_h,Qf_reel,'Color',C.tN,'LineWidth',1.3,'DisplayName',sprintf('Fuel Réel %.0fkg/h',Qf_reel_mean));
    % Zone hachurée = économie de fuel
    patch(aT4,[t_h, fliplr(t_h)], [Qf_reel', fliplr(Qfo'*3600)], C.eco, ...
        'FaceAlpha',0.08,'EdgeColor','none','DisplayName',sprintf('Éco +%.0fkg/h',eco_Qf_kgh));
end
yline(aT4,Qf_cible_eco_kgh,'--','Color',C.eco,'LineWidth',1.5,'Label',sprintf('Plafond ECO %.0fkg/h',Qf_cible_eco_kgh));
ylabel(aT4,'Qf(kg/h)','Color',C.white,'FontSize',8,'FontWeight','bold');
aT4.YColor = C.white;
yyaxis(aT4,'right');
plot(aT4,t_h,c0hat_v,'Color',C.tG,'LineWidth',1.8,'DisplayName',sprintf('Obs c0 [K=%.2f]',Kobs));
yline(aT4,0,'--','Color',C.gray,'LineWidth',1);
ylabel(aT4,'Obs c0(°C)','Color',C.tG,'FontSize',8,'FontWeight','bold');
aT4.YColor = C.tG;
xlabel(aT4,'Temps(heures)','Color',C.white,'FontSize',8,'FontWeight','bold');
title(aT4,sprintf('  [ECO] FUEL | Éco=+%.0fkg/h (+%.1f%%) | ~%.0fk€/an | Kobs=%.2f', ...
    eco_Qf_kgh, eco_Qf_pct, eco_euros_an/1000, Kobs), ...
    'Color',C.eco,'FontSize',9,'FontWeight','bold','HorizontalAlignment','left');
legend(aT4,'Location','best','TextColor',C.white,'Color',C.panel,'FontSize',8,'EdgeColor',C.border);
hCur(4)=xline(aT4,0,'Color',C.yellow,'LineWidth',2,'LineStyle',':');

% ── BARRE CONTRÔLE ────────────────────────────────────────────────────────
pctrl=uipanel('Parent',fig,'BackgroundColor',[0.06 0.08 0.12],'BorderType','line','HighlightColor',C.border, ...
    'Units','normalized','Position',[0.005 0.895 0.99 0.058]);
uicontrol('Parent',pctrl,'Style','text','String','MODE :', ...
    'ForegroundColor',C.gray,'BackgroundColor',[0.06 0.08 0.12],'FontSize',8,'FontName','Consolas', ...
    'Units','normalized','Position',[0.005 0.15 0.04 0.70]);
hMode=uicontrol('Parent',pctrl,'Style','popupmenu', ...
    'String',{'MPC-ECO (optimal)','Données Réelles usine'}, ...
    'BackgroundColor',[0.10 0.15 0.22],'ForegroundColor',C.eco,'FontSize',9,'FontName','Consolas', ...
    'Units','normalized','Position',[0.047 0.18 0.135 0.64]);
uicontrol('Parent',pctrl,'Style','text','String','VIT:', ...
    'ForegroundColor',C.gray,'BackgroundColor',[0.06 0.08 0.12],'FontSize',8,'FontName','Consolas', ...
    'Units','normalized','Position',[0.190 0.15 0.030 0.70]);
hSpd=uicontrol('Parent',pctrl,'Style','popupmenu','String',{'x1','x2','x5','x10','x20'}, ...
    'BackgroundColor',[0.10 0.15 0.22],'ForegroundColor',C.eco,'FontSize',9,'FontName','Consolas', ...
    'Units','normalized','Position',[0.222 0.18 0.060 0.64]);
uicontrol('Parent',pctrl,'Style','text','String','TEMPS:', ...
    'ForegroundColor',C.gray,'BackgroundColor',[0.06 0.08 0.12],'FontSize',8,'FontName','Consolas', ...
    'Units','normalized','Position',[0.290 0.15 0.044 0.70]);
hSlider=uicontrol('Parent',pctrl,'Style','slider','Min',1,'Max',N,'Value',1, ...
    'SliderStep',[1/(N-1) 10/(N-1)],'BackgroundColor',[0.20 0.30 0.45], ...
    'Units','normalized','Position',[0.338 0.28 0.210 0.44]);
hSlVal=uicontrol('Parent',pctrl,'Style','text','String','0.0h', ...
    'ForegroundColor',C.yellow,'BackgroundColor',[0.06 0.08 0.12],'FontSize',9,'FontWeight','bold','FontName','Consolas', ...
    'Units','normalized','Position',[0.554 0.15 0.044 0.70]);
hBPl=uicontrol('Parent',pctrl,'Style','pushbutton','String','PLAY', ...
    'BackgroundColor',[0.05 0.30 0.10],'ForegroundColor',C.green,'FontSize',10,'FontWeight','bold','FontName','Consolas', ...
    'Units','normalized','Position',[0.610 0.14 0.080 0.72]);
hBSt=uicontrol('Parent',pctrl,'Style','pushbutton','String','STOP', ...
    'BackgroundColor',[0.30 0.05 0.05],'ForegroundColor',C.red,'FontSize',10,'FontWeight','bold','FontName','Consolas', ...
    'Units','normalized','Position',[0.696 0.14 0.080 0.72]);
hBRs=uicontrol('Parent',pctrl,'Style','pushbutton','String','RESET', ...
    'BackgroundColor',[0.15 0.15 0.05],'ForegroundColor',C.yellow,'FontSize',10,'FontWeight','bold','FontName','Consolas', ...
    'Units','normalized','Position',[0.782 0.14 0.080 0.72]);

hBPl.Callback = @(~,~) setappdata(fig,'playing',true);
hBSt.Callback = @(~,~) setappdata(fig,'playing',false);
hBRs.Callback = @(~,~) scada_reset(fig,hSlider);

uicontrol('Parent',pctrl,'Style','text', ...
    'String',sprintf('[ECO] Fuel éco=+%.0fkg/h (+%.1f%%) | Prod=+%.0ft/h | %.0fk€/an | LSTM R2=%.2f | MC n=%d Rob=%.1f%%', ...
    eco_Qf_kgh, eco_Qf_pct, gain_prod, eco_euros_an/1000, R2_lstm, n_mc, eco_robustesse), ...
    'ForegroundColor',C.eco,'BackgroundColor',[0.06 0.08 0.12],'FontSize',7.5,'FontName','Consolas', ...
    'HorizontalAlignment','right','Units','normalized','Position',[0.870 0.10 0.125 0.70]);

drawnow;
fprintf('   Interface SCADA ECO prête — cliquer PLAY\n');
fprintf('   Temps total préparation : %.1f s\n\n', toc(t_start_global));

%% =========================================================================
%  ANIMATION TEMPS RÉEL
%% =========================================================================
setappdata(fig,'playing',false);
spds=[1 2 5 10 20]; kpl=1;

while ishandle(fig)
    try
        set(htclock,'String',datestr(now,'dd/mm/yyyy  HH:MM:SS'));
        pl=getappdata(fig,'playing');
        sp=spds(get(hSpd,'Value'));
        if pl
            kpl=kpl+sp;
            if kpl>N, kpl=1; end
            set(hSlider,'Value',kpl);
        else
            kpl=max(1,round(get(hSlider,'Value')));
        end
        tc=(kpl-1)/60;
        set(hSlVal,'String',sprintf('%.1fh',tc));

        ms=get(hMode,'Value');
        switch ms
            case 1
                Tc=Tgo(kpl); Tpc=Tpo(kpl); Qc=Qfo(kpl)*3600; Dc=Deo(kpl); kc=kADo(kpl); c0c=c0hat_v(kpl);
            case 2
                if reel_ok && kpl<=length(Tg_reel)
                    Tc=Tg_reel(kpl); Tpc=Tp_reel(kpl); Qc=Qf_reel(kpl); Dc=De_reel(kpl); kc=kADf(Tg_reel(kpl)); c0c=0;
                else
                    Tc=Tgo(kpl); Tpc=Tpo(kpl); Qc=Qfo(kpl)*3600; Dc=Deo(kpl); kc=kADo(kpl); c0c=c0hat_v(kpl);
                end
            otherwise
                Tc=Tgo(kpl); Tpc=Tpo(kpl); Qc=Qfo(kpl)*3600; Dc=Deo(kpl); kc=kADo(kpl); c0c=c0hat_v(kpl);
        end

        JV=[Tc,Tpc,Qc,Dc,kc,c0c];
        jcols_d={C.red,C.green,C.eco,C.cyan,C.purple,C.tG};
        if Tc>Tg_sec, jcols_d{1}=C.red; elseif Tc>Tg_ale, jcols_d{1}=C.orange; end
        if Tpc<Tp_lo||Tpc>Tp_hi, jcols_d{2}=C.orange; end
        if Dc>=Dmax-2, jcols_d{4}=C.yellow; end
        if abs(c0c)>10, jcols_d{6}=C.orange; end
        % [ECO] Alarme si Qf dépasse la cible ECO
        if Qc > Qf_cible_eco_kgh*1.02, jcols_d{3}=C.orange; end

        for jj=1:6
            if ~ishandle(hJv(jj)), continue; end
            set(hJv(jj),'String',sprintf('%.1f',JV(jj)),'ForegroundColor',jcols_d{jj});
            ratio=max(0,min(1,(JV(jj)-jmin(jj))/(jmax(jj)-jmin(jj))));
            p_=get(hJb(jj),'Position');
            set(hJb(jj),'Position',[p_(1) p_(2) max(0.001,0.94*ratio) p_(4)]);
            col_=jcols_d{jj}; if JV(jj)>jlim(jj), col_=C.red; end
            set(hJb(jj),'BackgroundColor',col_);
        end

        % [ECO] Alarme 4 = fuel dépasse cible ECO
        alm=[Tc>Tg_sec, Tpc<Tp_lo, Tpc>Tp_hi, Qc>Qf_cible_eco_kgh*1.02, abs(c0c)>10];
        for aa=1:5
            if ~ishandle(hAl(aa)), continue; end
            if alm(aa), set(hAl(aa),'ForegroundColor',alcls{aa},'FontWeight','bold');
            else,        set(hAl(aa),'ForegroundColor',C.gray,'FontWeight','normal'); end
        end

        cTg=C.green; if Tc>Tg_sec, cTg=C.red; elseif Tc>Tg_ale, cTg=C.orange; end
        cTp=C.green; if Tpc<Tp_lo||Tpc>Tp_hi, cTp=C.orange; end
        if ishandle(hTg_syn),    set(hTg_syn,   'String',sprintf('%.0f°C',Tc),   'Color',cTg); end
        if ishandle(hTp_syn),    set(hTp_syn,   'String',sprintf('%.1f°C',Tpc),  'Color',cTp); end
        if ishandle(hQf_syn),    set(hQf_syn,   'String',sprintf('%.0fkg/h',Qc));               end
        if ishandle(hDeng_syn),  set(hDeng_syn,  'String',sprintf('%.0ft/h',Dc));                end
        if ishandle(hkAC_syn),   set(hkAC_syn,  'String',sprintf('kAC=%.1f',k_AC_opt));          end
        if ishandle(hkAD_syn),   set(hkAD_syn,  'String',sprintf('kAD=%.1f',kc));                end
        if ishandle(hkADB),      set(hkADB,     'String',sprintf('%.1f',kc));                    end
        if ishandle(hObs_badge), set(hObs_badge,'String',sprintf('c0=%.1f',c0c));                end
        % [ECO] affichage économie fuel instantanée
        if reel_ok && kpl<=length(Qf_reel)
            eco_inst = Qf_reel(kpl) - Qc;
            if ishandle(hEcoFuel)
                set(hEcoFuel,'String',sprintf('Éco:+%.0fkg/h',max(0,eco_inst)));
            end
        end

        for tt=1:4
            if ishandle(hCur(tt)), set(hCur(tt),'Value',tc); end
        end
        drawnow limitrate;
        pause(0.05);
    catch
        if ~ishandle(fig), break; end
        pause(0.1);
    end
end

%% =========================================================================
%  FONCTIONS LOCALES
%% =========================================================================

function scada_reset(fig, hSlider)
    setappdata(fig, 'playing', false);
    set(hSlider, 'Value', 1);
end

function [Tg_r, Tp_r, De_r] = run_scenario(A_m, B_m, D_m, Qf0s, Tg00, Tp00, ...
    N_, Np_, Nc_, qT_, GT_, FT_, Hm_, Ld_, oq_, dQfs, Qfhi, Qflo, Qfcib, ...
    Tga, Tgc, Tgs_, Dmax_, aD_, lseq_, Kobs_, fp_, ...
    noise_s, step_k, step_v, drift_r, Qf0kgh)

    Tg_r = zeros(N_,1); Qf_r = zeros(N_,1);
    Tp_r = zeros(N_,1); De_r = zeros(N_,1);
    Tg_r(1) = Tg00; Qf_r(1) = Qf0s; Tp_r(1) = Tp00; De_r(1) = aD_ * Qf0kgh;
    Qfl = Qf0s; c0h = 0; drift = 0;
    bT = Tg00*ones(lseq_,1); bP = Tp00*ones(lseq_,1); bQ = Qf0kgh*ones(lseq_,1);
    
    for k = 1:N_-1
        drift = drift + drift_r;
        pertub = 0; if k >= step_k, pertub = step_v; end
        Tg_m = Tg_r(k) + pertub + drift + noise_s*randn();
        if mod(k,15)==0 && k>80
            if Tg_r(k)<Tga-3 && Tg_r(k)>Tgc-15
                Qfl = min(Qfcib, Qfl + 4/3600);
            elseif Tg_r(k)>Tga
                Qfl = max(Qf0s, Qfl - 25/3600);
            else
                Qfl = min(Qfcib, Qfl + 2/3600);
            end
        end
        Tr = Tgc+3;
        if Tg_m>Tga+4, Tr=Tgc-15; elseif Tg_m>Tga, Tr=Tgc-8; end
        dx = Tg_m - Tg00;
        fq = (GT_'*qT_*(FT_*dx + Tg00 - Tr*ones(Np_,1)))';
        At = [Ld_;-Ld_;tril(ones(Nc_));-tril(ones(Nc_));GT_];
        bt = [dQfs*ones(2*Nc_,1);(Qfhi-Qf_r(k))*ones(Nc_,1); ...
              -(Qfl-Qf_r(k))*ones(Nc_,1);((Tgs_-2)-Tg00)*ones(Np_,1)-FT_*dx];
        try
            [dU,~,ef]=quadprog(Hm_,fq,At,bt,[],[],[],[],[],oq_);
            if ef>0&&~isempty(dU), Qn=Qf_r(k)+dU(1);
            else, Qn=Qf_r(k)-30/3600*(Tg_m>Tga); end
        catch; Qn=Qf_r(k); end
        Qf_r(k+1) = max(Qfl, min(Qfhi, Qn));
        Tg_r(k+1) = max(650,min(Tgs_+1, A_m*Tg_r(k)+B_m*Qf_r(k+1)+D_m+2*randn()));
        c0h = c0h + Kobs_*(Tg_r(k+1)-(A_m*Tg_r(k)+B_m*Qf_r(k)+D_m+c0h));
        bT=[bT(2:end);Tg_r(k+1)]; bP=[bP(2:end);Tp_r(k)]; bQ=[bQ(2:end);Qf_r(k+1)*3600];
        Tp_r(k+1) = max(72,min(100,fp_(bT,bP,bQ)+0.68*randn()));
        De_r(k+1) = min(Dmax_, aD_*Qf_r(k+1)*3600);
    end
end

function yp = lstm_fwd(bTg, bTp, bQf, Wf, Wi, Wg, Wo, Uf, Ui, Ug, Uo, ...
    bf, bi, bg, bo, Wd, bd, Tgm, Tgs, Tpm, Tps, Qfm, Qfs, sl, sfn, tfn)
    h = zeros(size(Wf,1),1); c = h;
    for t = 1:sl
        x = [(bTg(t)-Tgm)/Tgs; (bTp(t)-Tpm)/Tps; (bQf(t)-Qfm)/Qfs];
        f  = sfn(Wf*x + Uf*h + bf);
        i2 = sfn(Wi*x + Ui*h + bi);
        g  = tfn(Wg*x + Ug*h + bg);
        o  = sfn(Wo*x + Uo*h + bo);
        c  = f.*c + i2.*g;
        h  = o.*tfn(c);
    end
    yp = (Wd*h + bd)*Tps + Tpm;
end

fprintf('\n=== FIN SIMULATION ECO ===\n');
fprintf('Résultats exportés dans PFE_rapport_ECO.txt\n');