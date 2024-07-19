# coding=utf-8

source_1 = """#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import math

class Network:
    WIDTH = 28
    HEIGHT = 28
    LABELS = 10

    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph = graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args, train_size):
        with self.session.graph.as_default():
            # TODO: Construct the network and training operation.
             # Inputs
            self.images = tf.placeholder(tf.float32, [None, self.WIDTH, self.HEIGHT, 1], name=\"images\")
            self.labels = tf.placeholder(tf.int64, [None], name=\"labels\")
            self.is_training = tf.placeholder(tf.bool, [], name=\"is_training\")

            # Computation
            flattened_images = tf.layers.flatten(self.images, name=\"flatten\")

            conv1 = tf.layers.conv2d(inputs=self.images, filters=10, kernel_size=[5, 5], 
                                                strides=2, padding=\"same\", activation=tf.nn.relu, name=\"conv1\")
            # conv2 = tf.layers.conv2d(inputs=conv1, filters=10, kernel_size=[3, 3], 
            #                                     strides=2, padding=\"same\", activation=tf.nn.relu, name=\"conv2\")
            #pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, name=\"pool1\")
            # conv3 = tf.layers.conv2d(inputs=conv2, filters=10, kernel_size=[3, 3], 
            #                                       strides=2, padding=\"same\", activation=tf.nn.relu, name=\"conv3\")
            # conv4 = tf.layers.conv2d(inputs=conv3, filters=10, kernel_size=[3, 3], 
            #                                     strides=2, padding=\"same\", activation=tf.nn.relu, name=\"conv4\")
            

            # flat1 = tf.layers.flatten(inputs=conv1, name = \"flat2\")
            # dense1 = tf.layers.dense(flat1, 300, activation=tf.nn.sigmoid, name=\"dense1\")

            # pool2 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2, 2], strides=2, name=\"pool2\")

            #dropout0 = tf.layers.dropout(conv3, rate=0.5, training=self.is_training, name=\"dropout0\")
            flat2 = tf.layers.flatten(inputs=conv1, name = \"flat2\")
            dense2 = tf.layers.dense(flat2, 500, activation=tf.nn.relu, name=\"dense2\")
            #dense3 = tf.layers.dense(dense2, 200, activation=tf.nn.sigmoid, name=\"dense3\")

            dropout = tf.layers.dropout(dense2, rate=0.5, training=self.is_training, name=\"dropout\")


            output_layer = tf.layers.dense(dropout, self.LABELS, activation=None, name=\"output_layer\")
            self.predictions = tf.argmax(output_layer, axis=1)

            # Training
            loss = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer, scope=\"loss\")
            global_step = tf.train.create_global_step()
            lr_base = 0.1
            lr_final = 0.01
            
            decay_rate=math.pow((lr_final/lr_base), (1.0/(args.epochs-1)))
            decay_steps = train_size / args.batch_size
            decayed_lr=tf.train.exponential_decay(lr_base, global_step, decay_steps, decay_rate, staircase=True)
            
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.08).minimize(loss, global_step=global_step, name=\"training\")
            self.training = optimizer


            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries[\"train\"] = [tf.contrib.summary.scalar(\"train/loss\", loss),
                                           tf.contrib.summary.scalar(\"train/accuracy\", self.accuracy)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in [\"dev\", \"test\"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + \"/loss\", loss),
                                               tf.contrib.summary.scalar(dataset + \"/accuracy\", self.accuracy)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)


    def train(self, images, labels):
        self.session.run([self.training, self.summaries[\"train\"]], {self.images: images, self.labels: labels, self.is_training:True})

    def evaluate(self, dataset, images, labels):
        return self.session.run([self.summaries[dataset],self.accuracy], {self.images: images, self.labels: labels, self.is_training:False})

    def predict(self, images):
        return self.session.run([self.predictions], {self.images: images, self.is_training:False})



if __name__ == \"__main__\":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(\"--batch_size\", default=50, type=int, help=\"Batch size.\")
    parser.add_argument(\"--epochs\", default=150, type=int, help=\"Number of epochs.\")
    parser.add_argument(\"--threads\", default=1, type=int, help=\"Maximum number of threads to use.\")
    args = parser.parse_args()

    # Create logdir name
    args.logdir = \"logs/{}-{}-{}\".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime(\"%Y-%m-%d_%H%M%S\"),
        \",\".join((\"{}={}\".format(re.sub(\"(.)[^_]*_?\", r\"\\1\", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists(\"logs\"): os.mkdir(\"logs\") # TF 1.6 will do this by itself

    # Load the data
    from tensorflow.examples.tutorials import mnist
    mnist = mnist.input_data.read_data_sets(\"mnist-gan\", reshape=False, seed=42,
                                            source_url=\"https://ufal.mff.cuni.cz/~straka/courses/npfl114/1718/mnist-gan/\")

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args, mnist.train.num_examples)

    # Train
    best = 0
    for i in range(args.epochs):
        if i%10==0:
            print(\"--------------\", i, \"/\", args.epochs, \"epochs\")

        while mnist.train.epochs_completed == i:
            images, labels = mnist.train.next_batch(args.batch_size)
            network.train(images, labels)

        result = network.evaluate(\"dev\", mnist.validation.images, mnist.validation.labels)
        if result[1] >= best:
            best = result[1]
            print(\"DEV: {:.2f}\".format(result[1]*100), \" Best!!!\")
        else:
            print(\"DEV: {:.2f}\".format(result[1]*100))
        

    # TODO: Compute test_labels, as numbers 0-9, corresponding to mnist.test.images
    predict = network.predict(mnist.test.images)
    test_labels = predict[0]
    with open(\"mnist_competition_test.txt\", \"w\") as test_file:
        for label in test_labels:
            test_file.write(str(label)+'\\n')

    # BEST: 99.84%
    """

test_data = b'{Wp48S^xk9=GL@E0stWa761SMbT8$j;B_w+VO;<j13gV=7Yq_8Jt(?mHZYog4pqV6G^cyK9Szyu=and!)1{nG33dLhx%)xq>_3>kdU~Z`F4T0>95dS@Bt*~M#$9)r04^ELbYe$?aKkIP4K>l)Tl4^S@eXM@|Jtbtic9#`h<8qd^wIJb=>PR(G-HT}xR5GV7&Av@*1rf4%tx&s21;kT{mYaDom40&ilyk(9uU}cYCqh4DQcttpn81jw?w=>m%<NAzIbRpF}&=LaQnl4y2CKc#$KHOS@~Aez?;kn>SI8dzA<%lR4vHGpkVhnH;kGeJo33`Vp{e66|hm#kv8V-)?JyoZGu`d*m6~ajYBuWj#PiH{(IEa<^E7oSQdX)?JEfaC2`m*20xO4BHW@+zSfKX{O-N~lUe3m<+co1t{!676e^b{H|+foL-Aa7x~I?qwLEL_10vPO6;wleQm^s{Rvgc<3zOOMHqt<OnbqZVDTW^#`r;ZV(c69$v2dOV3A71nA4{&g6&mp$-O9~ct}-pg77AS-UaNPG9&E9Di{}RN%eTz#B1iST3+hoMSU5k-v$uOpR}4Q=k$H>1XtkoR0fHKwcxE=d0KG@j<<|Tvy!=E2)BY;#;ctF|l+x=db`K|IzjBR^@!QkQhYh$&=N?VS*RYxuB@zmER_@Qp>feKQn|QX5-#LQk-#W|qMH1jT2CneLC&C1}-uSsV=5ibR<h*b+L|yC%m3`95ZXzvvt-&!+eVTK6MZ)(eSwLq2O<?!Oy-PjPdc>_k7I3}`PHx48!X97;G;CCH_Ry8*84?~<V)kkh_yg}X?s<hwj<xS9@|_;$tit!SSbP+>%X^AG(q@`7c^jj}$rc5^ZVJnO3M7+ce&8&i+E?s*9}antV0l)XCT$3+Pdn;H2x%<zyhawuTqH+#==j9zTbU{vE!w)ht(U^R=thToDj9O>iB%@kZFpvep4)6bX|4Jz%NOz1;lgpHF#~~XCBR8hi_T1MegWRu=aQ7gCB4VH0S};V1lKov5CYCA9HNc*8#3MUmg)>q{y!{Kkg%{^$pRs-$KWEBeK?@cmvivM-7{Nlutrx_UJ;1)SXp|l$e81bu5VGomfOQ%iG$;R`m<RLJ|6r>N~0m+PmJzhoUv`r3iI5LH}kGaaa6g*--PLXkG1y4mI6Ov2adYUJS4x0q8vUkmNneGSmBWK*!m%AC5vRtK5le+GOSt)x!_XeWjwz!OoLa-IR}%kV@u2;rMcjf%xC3s?Xtw3(7qx&GAIQ0d6qZGg5c6J%FIkI!Q&mWQY<e2W2&LqF^arvW%W9$;0#Psn%A*U!xkY7(yaRh08?Aj2xK!50=UDcFo>!Cc0Rxm?q0gq@&%F?9z~sKD6rUMXuh8RKg}%1+f>7mwSa#x`o<_7@c7{#f>!@q<vb^i)k6KbwWqDOg<dOqI^qlJYN=^K6sA*3zS!uy`$apDMO@%_fF8e}1BX8I=$*O`gr7^n;`EP7*&NK73=Jm^b7F^S=xYCMzdO~I;sb|Gmmf-StSs7o=)1m!yatKUB;|&vW0L|YY4Pg<XS(8-edx0`h=Lir0b|iRS#LM;=hljG_SxoZUcOb{OgNsqREMutAKS1`2{}(u{XqT*B&64$H;MHr3{>NTf-y$q+)epRn;sOOxOJ(DyJN#P5Zsdcimf+oBz@Kc!Qz<myL=B;i7%n@#4*fB>FC-CTvB&FK++C8clY6o>D%7cxnS5+i$xT^%9woMxOC`ln(0ko!6;-I#a!ZPnGQH3S^)FxmJ5F3$2&7%g-@T1<CJ(0(dh_9OQHQ;#dWuiU=l!yRTJ@}UnKL^1a9Xm55`P$v65%-?h00M7(&oP>oZ>_*KIue9x1X+O+e<c|G*_KD4ft|`gyKflgJp0LV3?%9k?=u2%i`Mpa>0@<h3y3;tbic4HFK?$Q9|;g7MO%Y5wX@L`3!D^b-Yula=vIZmc$kT=!jwEyU<Mp_o{=@*p~(0wD9qfeR@Z;W<0|Q`h1;zbd3If{UgPsyCPYV_r_e^S?iykg(qKn`RohRTE4>P%5CzT}{HWu|dbUWJ{lT^CCSh?E124Ul`IoGV9~~lR@vB-A!-q$P@Yx&cL=LF>1)hv;4%^FLr?Jtl#(Eo;?ox;oi9uiqn6X0^)-K{mYa`BDksu(eL{u0yb>4BF6CG`FLQcGJY^oUi`e>cZ|~!|I61yReiOV!hAXe@*n~_s2R^4-<Oc6CRJpWCHn&NQG*}sjTMExi$CqtF^5UJQ2qkew=Y*QsaYMOH>&spX6hG}$Wb0Y?_1PU4+%hw^`gLJK>b%L(#Hyu4#Q5wwP&uFElL%ZK#azTZE?CmEtOE<zTrnNGtdl3e`6w(UWq=|kBY`+?I8;k(DRHDt7&kAdY9X3HFtUJ?%YJ{R}w2|+3#-RHcV{5xSo|Qeq!ozXZRGN;cP2RzpnSJ=Ng!c+eZ4kgt2CdQWv#l^QIya>$MReRSx}E9954jR{N<lz9od5#%Xf>M^QAtCznymg^BA;l&OA3%xY(a%QC*JRn)7GSJv1W>k1&i*92m~vUHMe!1J!B&S~zVpPcEWK;xeZQiA(=De=&WcxR>|CguoKIj}fMU6OC6b#kVe)M*8yi>%Oncbx8SO+t0wR`yPyKszxxsWGJ~dImPUqGqK75S-$*Ss=Q9co!B*mh6@bYG>4rk%ffPA0DI7of3l@VZ6R_N$COv>17ofFB2n08|tdim)@AtU~zDAwVKPbuSl4`Q$YB1p!Q^c?3K;BCwN<BZLQi#rP#1v?kg?)%(z{Px{ZaQw(u$>upskzip9m>m#%teOE=4|uFpf=Q&ju>GK-4yv^e>yAP}`{Zf^r?owZ#!(3PWkT`+4%0Q%RM>|CHOA`9c%J1@NPmd`oU98CssEHuec@NQT@<D3{YKAhY}RgB+fbhJ&RvSd}l+doY2dpd?Bm&5G7MKz^eE`{0kra9h4`g;yg$9v;Sz7@~;C4laS$oj9#OW_du1`zpK>Q$?83eY`PkOuDy(azGIs}&?>tEP-T#AE@6SB&a6^8}_NT=cPg&$)5oEm5O@8(XYr3TMl%K}q8+bw=;loV=BYU9L!4*-(ZZbc!?Lv8{g0wFUZPaciIvWGvV0n$6Q})G#hn_aC~!hjx5<Fk_4@!LILX^Q!vFGfNezq*y)-IWM2ENnLXoKw^PNZ&ebLY9`v^@|yO-9?V3nOcDOCyvL4nCo|OxA|<~@=(JMf1??}h?|{MYOT3EIs~Msv>(1@Jebj7D3C)~C9fG;eQ930HDTH~txhZ;fdw-j#Dn@`2amHN1q8WPMv+9Qwozh-sFM+01+^hgztzka)v+6R1U(RyjVnLc?ZlHdu&GavSz9U-pMasDTDiZUPbe+dnG^8Cixk`4yS+W?fGrT3Br6DG~e0<)$TTwir=N}nkZ5nFxoBE3F^~XXm(Jfx{<6v4+gkz4>tX%r@VCbRe3a%3N972|nr~|ua=>*cRNidPO*5G^|rgIN+XU~Au_QJPWCD8f{&~BT??hL3Rl?M)E8||!1rr3o;J?~l~Z{BQq1jH)wR;d<%P_9xOjT9*Xr#)<UT%qt7S=5<T2|rQ(9XaaL$SLkTXC|10IN_vW34~n_c(>8ab<J`P@C6PzO|m)}yjp@42in;h6i9wWybuqb3faHe(={S!XeQc+V$oA)dAp}C*rybQ7Fu)?*OJjK0s?P7?wWO(-dvoKPiKoRH{AA=Wc?k)?bWwFh+8C!C{|0n#6SsE+1Ql~AO2c1sF8yR&xwE9lrVbQa2%EBf@!op(Vs~^Wy_?p<eR5R{L1iO*&StgWUqr046mX^@Eq6Z0;q|#9YDLo_H<kHnv1I^O+Z>}SB!tK03C-EguoWm`hE<v>s^Ap66}H=H))BO`@MICv~L_8?(K4DCCcirwYZKHtH{alIV;W-M|%{)DEp&NX0=U0{Lej+-!{N?4Sr4iK@wQC_qu+o9em+3etI~;hE>gJ^OW3Rf+dPIcfSo=3*c?GfeC~I;pJdob}1c_BkiVSLq)!JJtGEIPyzWx@DMfceR37s^7JY@!#?4z3yzB28qU5AhT8v7Y>)z`@nMl7g!n)iP=P?tqX20V=02xuJVpCBn}Wqk*cKYdUvk=EY;fn(vYn-*1VCd4p^J-cU{x)UZzVe``cWsc37}^<=OBM)YARtRX!nAAM%qJ#@5Jb}AM?ZX!Sj~#4|`|d2M+@EOcla+$+m9L`1KMalb=+EYQ&o(rpAzAh*o9Z-FdgQ77*|;pSvCp;hlvm#+~qx7Y7UvdI4^Xhd)H+*?j)wFGb<Q8~gFE*-*RSG0Dmm3(W!?V&JEfS>EL8c)toXnQm-4t~qPUW5ct=tc8I3UY1v9xN;gYt3tbOZn|$r39Ui5kZ?6{V>}40^hGVYW@dOo^x)V8G4izEfsdHh1yPJO-)fK$b(PG?417{&I`ct7JOa%9s`=^P^!Gh1Xm&>b)Boh6F48ewQTcA1jGcop_{dT$pA&6PU3y!uqbLHg7fQtfEPB!8nZv70DVF>QK^y);5w2YidLUK0jit$uA<yIFt<<xZVHTq~w)eCS12~_@N7!Rl&VX}de*(SJprDNwQmm{Qb-pPq6Y@jPd{X!<YKzpFwcfWi$>?=Vr?t)rx~^Ntt&bj~AhrC#LsOKxHfKE>-qw(GLJHtR?>^UX_N<W~t2!N*RMcD{O$+s7YMxEzfupb%m1(t`Vx%Bf3SS`TWvBqS9y(9PIOC<`Dp%6|J9JH{fJQU*F$<lm1L05X4G{a^{2)XZ{$uC)GQ%#;B^JDf3w0Xi1e&R5|9rvbT{_3YuC8{wG_Uc9zi0D-$@zTTnqnyuaHNPtV(EDj=-f&wWmJBY`udVbwd$QCC{LZW_PZC2hi#HpDA;(q&b22_$gi5FYIRT6fj~Uh$vSeiv`7@v^1J3uOke3j#3+mGi5*ovzz9t=Iv2r}e|+eXSR0du?!U;+8XHIve2@1StUJno>VOb61-tEK!mx^m`#+vPe#RwIR`JFu#n%<wbLgL?YVa*gja#|ML-)|A%8%}gjh(2R@}8KtXPpXikf$cANM1-wDXBMxc>o;MlkVP^+<=uYt#Wx8JL_v6&X9Z}pWC!dKh5&ApVMCn7oepZ!2YAjhmQCFE8q2D<!|#pa+~g-BfB*pQd*e;naUFZ2C(YsnSh7>nJwA&=K|JO|G?ROV(?8DKcmBN8_tt>iUCDK^0BnlZRTKyT64xfZyIkYjZw1vC^tf>z*VJ>-1jfvF<8c{x`;duq>nNK5YhtIKdfFNsEy~qCQjyFFtqMe{QA8wEfO*&?Qh+8)+oU)xr>n#L#O<p<QNY=-O1)Rn9c5%Is<ZLo;G&A&K{zynwRC1KKK+BxkS(eJ&<I)4X2~`<9)V{Jq4|Ecd~-+NhdYwul%w7jL|v52i_-Abg8gT?-z6U-w&DGwCRg9Fgi<M6NwlnAg#TQC}=5$V^<95US~n?{L1Plmafr6^o}y}Af%U5W2%Mo@=*{1k2CV6`7F;UH*XlA1?8#~m&G4Nx*4q08*?j3dmSm*M1#18{V@PS+e@Ow1M#q`{;Km+_-}o_layr&WZ>cT9615Su=MBD@28ho-0F|q@0jLw*uO#IA={{%hA+S;$xiOtIq`pC`EzX2efFrD64}Q`+7vB8P|#iP8f&3V<BP|RM`}ZW0%U*)63larO+M1EKH9eeDy~DVb8GXPZWd1j@HZ{+V6Q#PtGJO>DRV~dfMZ|dIvUp(7<Bm-L#${u_r(Ze4d<N(2O5ij$J;k8z%5W)M_Ua5)5YowDrp|rTt_yi+ra@$Vr4p#gxKwZ|6lMq;DL6>gh8xrG$587kD}a5$A@<NX=<%i%-7QB@}Qxb32xkpai^{xcj=&BMf#&)2jYOR)e(1)GfQFBg;0|wF7$X(skUt@sT(dziR7tN=;NR|`)o;<DbW5#qnnoo*$hc3!n;hq?)%D_FuVPoP4c$o?NWi2L!!tj=6Ag_=3M4E6Q?#7Rz|m<A5>M$uwNQzNrPPPdTTUuTdjRBIY&~(gtiEjPj&c5-?jLb;~A-X06*W{g@Yazs*i2})eB5tQd59MV8{|P4RYUq>gqg-WRn&}d-|()^~Hoe1VgX7aPBFu+5G^n5b;l52Y}JzgTbAQA<UbU;2kTPy4P@O^1Vmo#L>R(Bq7?ZC75Q8DdjkJR*k~IvMKOb&*HIip|kR=2GG_a{i=`6)QI`9r881TiM!2h`2C>3RUXm3U6)<tZG}et_k<pZIpvD6-p?Fz4ya}qQ%y}vTn@ERH-T3*=Hco*;iy7zgK|;aTVY<2mT`QX`*jYT302B0TvSklNqa=pYGBRBT9C(yriCS9MJt<QgL^&<Rcny?b<QT?`H#r07c@oj1E-EX{-K2maF;jt3ZlJ`zdtVncq!?oI_m5A4sFG)_I<1Ogo<HRF-72z1A-HbRp)kEL78|KhyE*OGim$M=O3=iM263inVXPyv&QI{pkNA}Vj!<(1t=4=Vf!*j5HqYj_P=UVxr*WczMpa8a2*_;+8tf&LLU+e*FgVS(HM;;L8J!n+tSdn!FKgCe*v_aKSD@hMa1D@2`rophJWo9JX)M8=2Ww!Oi=UfCHNNRHj5MIDg9bKTgh8n>9;&_!YWEf^lGXgJv`M2@wbBXMSXeCmK-dD)Cf0d-84jFF!+5Ne=BSH8g$)+TeUJJf$>TpoBw4nJnt~ahLTs8^>eDyF^sK<K4$o#q`~^DjFNZNfph$p3B>2$(n|U3EwzKuWdz25Rj-Mia|q3D|LLL|(FGb-$nca?(&N%Hi@#WcH3R8H*55iA(&yj{a-D8ow!`}m&=Po+pWPaUq7qgmZmje*FwHvFm+Kn(Hc&if;e9-r?w}VCzkLkeQt>reWePvv_J#9<-196Xcmv&_J^dqfvVDTA2O1fs+UfUdE`GH6xjGuMg6!}%OxWD5OANZ(72UdjZ}Pk)5r$_{g>KJ%5hYj}KPjP9)MPw0eAWjwUN8J0B82T6T`KS^*lsw7kb}7U>7E$>SrnKfEVVdO*QB!(SUI^p5kFJ6Z%=V(#XfkOk?9R5446FxAWx~P*B?w`6>9yUeW<8%^}QE_bLB&r^Q!%09h9r(IZ<TsY}L~igcZi0$7wAKj^A9ABXt4(6b4I|4qb>Qz6LHNa``k)As9=ML?G`5WmZSJw{Gg6x-W(JX=QM!xt>p5c_l(K{Q^L%PFsLMexDS!gVP7y%+DwcoqCa8fjSVwo@CnFZp>q>l-Q$I<M!^5M74AeKYw`2cTsCpj0Z44(+F5QS5K{e%7a-)i3}2?1fxr!oL_qj{b}^?Tb!URJ{)4NhTe6gq`Bb2_X9ccKo<V}+gpV432w<5v6yIWT*(P09e=f6=Ai1Vs`@iMUsnNqllNgggiEZ`t<Bac00000uJ_wdA9fO}00I3hu<8K-vNm+ivBYQl0ssI200dcD'

if __name__ == "__main__":
    import base64
    import io
    import lzma
    import sys
    with io.BytesIO(base64.b85decode(test_data)) as lzma_data:
        with lzma.open(lzma_data, "r") as lzma_file:
            sys.stdout.buffer.write(lzma_file.read())
