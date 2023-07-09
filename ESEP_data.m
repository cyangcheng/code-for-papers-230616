clear all
clc
dem = [11833.30566	11579.16211	11256.5	11128.05176	10928.22168	10784.39551	10655.9209	10611.86523	10385.625	10291.83691	10209.94824	10162.99805	10127.15234	10055.97852	10140.95215	10108.91699	9986.808594	10027.19238	9916.37793	10040.93262	10044.37793	10130.95313	10138.42285	10380.83984	10665.41895	11126.4668	11441.52051	12083.95313	12399.93457	12890.17773	13285.07129	13751.57031	14285.23438	14929.39356	15500.50879	16042.66504	16625.07617	17062.23438	17346.5332	17556.09961	17657.63281	17784.81641	18008.84375	18112.66016	18342.08789	18538.0625	18335.61523	18111.33789	17277.78125	17040.91992	17125.59961	17288.59766	17346.20117	17374.38672	17312.03516	17106.88477	17086.47852	17053.73047	16998.36719	16823.79883	16809.56641	16726.91602	16627.81641	16529.06055	16567.17578	16556.22266	16716.79688	16518.02148	16596.48242	16594.13867	16481.61133	16304.2793	16352.27734	16591.83594	16860.47266	17088.625	17209.21484	17198.96875	17334.1875	17362.55859	17378.45117	17444.99219	17467.92383	17480.53125	17273.89844	17023.64844	16708.39258	16259.32324	15713.80273	15179.2832	14639.6543	14253.23828	13839.2627	13638.76172	13213.74414	12817.67773
10536.18457	10318.88965	10141.37891	9869.981445	9751.269531	9640.345703	9477.46582	9387.625977	9251.470703	9247.06543	9090.021484	9063.807617	8990.676758	8939.466797	8934.746094	8850.126953	8902.758789	8863.283203	8857.727539	8857.274414	8882.648438	8919.741211	9108.144531	9130.619141	9343.641602	9572.969727	9750.768555	10097.80566	10315.19922	10804.9209	11194.62793	11860.79981	12388.7666	12902.24902	13434.13574	13938.00293	14333.38965	14654.79688	14908.93164	15001.21875	15090.08008	15161.08984	15371.18945	15489.30469	15486.38867	15669.82129	15570.45215	15215.1377	14561.66504	14339.49023	14330.92383	14169.81934	14273.08008	14326.44824	14263.99023	14082.99219	14095.85938	13984.84863	13866.49414	13724.68066	13526.39063	13611.42773	13503.10059	13558.91016	13387.34082	13596.1875	13740.77734	13675.88086	13867.16797	13797.06836	13713.32227	13663.22656	13690.06836	13707.64844	13946.78418	14128.97656	14177.74316	14176.92578	14195.33496	14258.93555	14291.83984	14408.74902	14524.50195	14478.18164	14384.92188	14314.09473	14193.06934	13956.85938	13546.63281	13096.10645	12719.20898	12372.41309	12035.16895	11877.22754	11584.15332	11251.39941
9156.569336	8995.001953	8853.892578	8703.269531	8600.717773	8538.185547	8458.908203	8316.005859	8177.556641	8271.765625	8195.797852	8129.014648	8100.729492	8095.103516	8022.979492	8054.098633	7957.432129	7973.689941	7959.281738	8005.056152	8023.88623	8032.442871	8206.330078	8287.980469	8501.21582	8890.982422	9090.760742	9606.025391	9902.908203	10239.70703	10586.15625	10861.90918	11283.42969	11593.82031	11985.46875	12373.84277	12717.77637	12957.91504	13115.48242	13279.60547	13396.72168	13540.22754	13683.37402	13776.31055	13833.86231	13997.56738	13791.15723	13484.89648	12824.2207	12563.0918	12611.34863	12609.05762	12849.32715	12873.8457	12941.37402	12839.44922	12861.41895	12808.6875	12749.2627	12704.51367	12613.60156	12699.14356	12614.17578	12645.34961	12809.61523	12863.56641	12981.77246	13019.42676	13084.58594	12929.14356	12608.04883	12721.49707	12607.92188	12497.24902	12531.51563	12611.59961	12668.60938	12717.09668	12742.39941	12702.30957	12566.35254	12667.7959	12703.22852	12574.38867	12460.57129	12309.41895	11964.09766	11702.6416	11314.52539	10994.05176	10668.03125	10523.44629	10275.18066	10191.35742	10029.80957	9669.75
8899.731445	8803.902344	8610.896484	8492.222656	8458.285156	8364.361328	8286.467773	8290.323242	8163.617676	8150.983398	8084.762695	8039.863281	7944.605469	7952.231934	7881.595215	7904.224609	7915.572754	7852.754883	7951.150391	7986.285645	8009.961914	8161.408203	8139.511719	8275.979492	8722.349609	9113.894531	9567.200195	10018.00488	10077.3291	10297.00977	10521.77148	10704.37402	10953.64648	11256.88379	11606.69238	11801.45898	12071.85742	12333.5	12433.27539	12603.15918	12656.5957	12723.61914	12996.25293	13131.47949	13165.44141	13335.99023	13150.65234	12868.86523	12144.03809	11916.44043	11964.11719	12164.50586	12246.24707	12402.69141	12371.80859	12484.16406	12503.60645	12485.05762	12594.44727	12361.4707	12378.1416	12313.34766	12478.31445	12458.75098	12599.50391	12734.53223	12795.7168	12932.0293	12982.81055	12948.24316	12990.60156	12723.45703	12638.77344	12573.21094	12497.70606	12523.78125	12347.63867	12479.96875	12574.97461	12517.36133	12365.62305	12164.60059	12114.42188	12063.48535	11778.78418	11537.65625	11411.125	11153.3418	10696.68066	10472.3252	10207.31738	10040.96094	9871.557617	9876.289062	9683.34082	9504.09668
9177.947266	9150.412109	9039.970703	8881.508789	8827.993164	8625.246094	8646.599609	8519.273438	8460.770508	8441.583008	8352.160156	8249.983398	8253.795898	8198.824219	8227.415039	8272.923828	8167.041992	8138.508789	8222.255859	8294.294922	8392.702148	8437.111328	8597.095703	9007.832031	9208.212891	9580.264648	9770.947266	9893.648438	10008.9375	10191.80176	10284.96875	10542.39453	10883.85449	11162.41602	11490.95703	11891.43359	12065.86719	12486.65039	12525.08496	12766.03711	12822.58106	12903.83789	13123.9043	13150.0332	13337.24023	13619.40527	13486.08789	13065.20898	12615.6875	12583.19238	12705.11621	12752.61133	12826.52246	13039.55762	13031.05273	13006.22363	13142.13672	13218.91797	13280.2959	13356.61914	13387.53125	13439.45508	13293.55566	13514.45117	13469.12207	13573.64844	13628.53906	13641.4707	13704.35352	13737.13965	13533.50391	13343.73828	13243.22949	12944.51074	13081.51856	12903.21387	12827.88184	12589.67383	12551.00977	12562.55664	12692.79199	12881.20606	12823.26856	12697.97656	12625.46094	12494.73047	12238.04883	12018.21484	11712.3125	11487.25684	11210.26074	11024.02637	10968.37891	10990.26856	10815.02148	10557.9707
13734.90234	13490.18359	13341.17969	13077.32617	12897.34473	12790.70898	12491.2168	12416.19238	12220.5	12119.14844	12032.3418	11889.86914	11729.93555	11725.59473	11499.35645	11520.60547	11421.71094	11285.39258	11330.02734	11262.63965	11255.98145	11340.5459	11494.66699	11498.00684	11632.27051	11818.08496	11923.49707	12111.12012	12200.14941	12389.98242	12691.06836	13103.87305	13606.70117	14434.55566	14981.30566	15371.60254	15976.25781	16484.85742	16736.30859	17160.01758	17540.11914	17832.24805	18096.99805	18157.89258	18458.3457	18738.92578	18739.72461	18739.17773	18347.47852	18371.62109	18606.95117	18874.17578	19097.85156	19207.10742	19148.87305	19139.24609	19139.07422	19157.07422	19234.57227	19190.75195	19116.80273	19176.14844	19085.43359	19155.94531	19021.60547	19036.91211	19200.86328	19170.63477	19238.625	19145.84766	19018.95508	18888.84375	18783.11328	18711.09961	18669.29688	18668.56445	18509.83398	18450.61133	18251.68359	18297.35938	18488	18829.11328	19216.37109	19314.58984	19643.76953	19636.70703	19563.09375	19467.26367	19203.55859	18731.66016	18288.18164	18110.77148	17684.01563	17635.49805	17171.38086	16923.64063
12686.43066	12526.42969	12286.86426	12193.98633	11962.02148	11803.51172	11630.93359	11480.80078	11290.11328	11240.94824	11065.30371	10924.90137	10932.84766	10786.76465	10763.26758	10663.37207	10632.87402	10501.81445	10490.55273	10459.54395	10452.42773	10461.16016	10585.35059	10671.34766	10784.36621	10869.99414	10930.97168	11223.94922	11472.29981	11637.10352	11946.80762	12374.05371	13044.4707	13592.71777	14265.12207	14807.06836	15320.93652	15956.13672	16253.01953	16583.77539	16878.65625	17148.94531	17539.30859	17726.13477	17972.83008	18337.46289	18407.36523	18243.60742	18044.87109	18110.17969	18393.13867	18583.37109	18672.18164	18924.39453	18882.60742	19021.34961	19082.52148	19148.7168	19136.07031	19281.77148	19173.72266	19299.54883	19357.11719	19321.04102	19336.4082	19461.80273	19484.15234	19605.81836	19552.78516	19453.80469	19282.65039	19039.46289	18822.94922	18636.0918	18786.87891	18571.95508	18432.60547	18177.28711	18029.17969	17934.03711	18016.32227	18236.38867	18735.79297	18840.39063	19074.36328	19290.97461	19267.75	19092.82813	18862.19336	18513.39648	18277.28516	17840.88281	17787.62891	17547.76172	17184.30078	16701.84766
17914.88281	17498.70703	17146.22461	16828.68555	16595.03906	16211.60352	16080.7334	15822.2207	15650.82031	15330.1875	15104.05078	14989.51172	14815.74023	14657.68555	14499.50195	14417.07422	14234.32617	14108.82422	13982.73828	13937.68555	13796.46387	13810.26758	13801.43555	13690.76953	13756.33203	13657.11133	13768.08008	13770.64648	13841.48828	14128.32227	14485.99219	15029.15234	15491.39648	16311.27148	16929.60938	17599.47266	18385.4375	18853.53125	19422.81641	19902.13477	20351.62305	20772.0625	21217.0625	21644.90234	21977.90625	22470.70508	22626.77734	22776.0625	22623.22656	22923.20703	23287.76953	23504.45703	23765.47461	23728.93945	23685.2207	23668.78711	23655.71289	23654.29297	23546.80078	23427.42773	23421.49609	23438.64063	23419.11133	23373.26563	23214.51563	23301.375	23049.77148	23245.53711	23154.74023	23000.4375	22695.61914	22431.50781	22113.90234	21896.44141	21880.12695	21527.83984	21391.59766	21112.71875	20979.79688	20878.7168	20997.58789	21381.38477	21743.46289	22046.73828	22333.03711	22443.25195	22378.28125	22237.2832	22048.01172	21765.46094	21505.79883	21145.05078	20767.65625	20603.13672	20143.58398	19713.14844
9294.266602	9204.09375	9121.554688	8942.848633	8723.660156	8722.724609	8528.095703	8600.516602	8391.824219	8373.363281	8347.867188	8278.027344	8201.537109	8146.17627	8146.666016	8180.128906	8090.882812	8112.768555	7999.027344	8171.126953	8122.918945	8345.790039	8440.322266	8591.223633	8715.208008	9188.060547	9467.338867	9792.416992	9948.369141	10180.70703	10430.66504	10831.25977	11351.5957	11925.45898	12439.26563	12795.1875	13146.29297	13823.37695	14125.0625	14386.18945	14674.96094	14944.02734	15133.74023	15299.29883	15604.34473	15824.33984	15841.71094	15491.91406	15239.3125	15256.02148	15416.82227	15675.93457	15713.68164	15866.91016	15860.26563	15820.38086	15948.25391	16179.60449	16271.04688	16405.94336	16353.71484	16600.5	16662.78711	16738.83008	16723.63867	16727.6875	16768.68945	16942.05664	16746.39453	16727.37109	16514.27344	16295.20215	15982.13672	15884.31055	15762.80469	15701.46875	15636.11719	15915.51465	15832.31641	15617.10938	15472.92285	15341.97852	15279.12305	15109.04688	15005.83203	14861.4043	14752.70898	14443.99219	14139.75391	13774.87891	13517.2041	13187.02832	13082.94336	13116.05078	12944.75	12739.26172
14059.9502	13936.76367	13896.00391	13583.02637	13363.06738	13257.74121	13093.62305	12961.60742	12828.03125	12663.98438	12497.41602	12359.27539	12157.01953	12147.17188	11941.44336	11847.22656	11760.2959	11706.76563	11556.14648	11520.90918	11555.24805	11542.92871	11556.79883	11557.47656	11598.82227	11659.13477	11857.36328	11860.63379	11935.47559	12101.94922	12220.29199	12612.28906	12957.7627	13201.87012	13554.11133	14001.95215	14414.76563	14948.88672	15393.10059	15770.88086	16085.32617	16473.59961	16779.76563	16941.65625	17248.12695	17498.31836	17535.7832	17615.14453	17572.92773	17587.20898	17839.34961	18061.87695	18284.79297	18514.81641	18652.44141	18669.38086	18736.98047	18765.10156	18826.69141	18808.76563	18684.97266	18709.61719	18667.50977	18610.87305	18584.55273	18393.71289	18226.41797	18258.625	18121.16797	18071.98438	17808.19336	17559.91992	17406.06055	17131.01758	17114.73633	17011.74414	17298.60938	17359.4043	17284.96875	17334.63672	17217.92969	17395.51758	17452.77344	17417.23828	17461.76563	17237.83008	17175.94141	16987.63672	16820.25	16457.07227	16288.20703	16153.35156	15942.0918	15919.5459	15746.88379	15425.08398
10354.63086	10276.50977	9976.861328	9984.733398	9801.276367	9664.31543	9632.862305	9511.644531	9430.126953	9284.78418	9267.916992	9187.914062	9217.619141	9232.765625	9157.189453	9163.089844	9151.009766	9081.113281	9061.275391	9156.770508	9190.990234	9253.423828	9518.268555	9661.920898	9827.678711	10319.56543	10649.29004	11084.58984	11363.27539	11831.23731	12075.71289	12291.28223	12769.10449	13178.57324	13706.87988	13973.52148	14367.92969	14768.71191	14788.73145	15047.66602	15167.31738	15370.13379	15563.31738	15608.81641	15658.92969	15781.29688	15695.27441	15410.59277	14698.24316	14560.21191	14736.88574	14831.29297	14939.49219	15269.04102	15249.92676	15154.10156	15207.6084	15293.2666	15064.91504	15170.15039	15094.23438	15066.94141	15106.03027	15203.48535	15111.35449	15230.85059	15363.61231	15553.20801	15328.16406	15324.86231	15132.36523	15073.9541	14998.18945	15104.42578	15027.32813	14955.52051	14756.12402	14654.49902	14604.21484	14490.00098	14410.7207	14386.37305	14331.22461	14303.6416	13974.14844	13770.05957	13546.16211	13267.18262	12784.29395	12385.83984	12102.67383	11787.93555	11654.26367	11714.35742	11460.93457	11236.64356
11767.2207	11612.45996	11363.75684	11145.52637	11013.94141	10824.84473	10681.98535	10559.10938	10430.50293	10329.90918	10279.74121	10205.77637	10222.75488	10178.01563	10082.77734	10094.19141	10061.30273	10030.11523	10056.66309	10015.30762	10159.75391	10219.37402	10280.93359	10535.49609	10681.31738	11221.35547	11554.44141	12092.08691	12340.49414	12782.75879	13176.9082	13525.31445	14025.81543	14560.70117	15301.11133	15910.03809	16411.73438	16975.19336	17170.1543	17398.93945	17646.32422	17835.89844	18005.17578	18142.625	18323.09961	18453.20117	18475.62305	17997.21094	17366.62695	17263.91016	17317.63281	17393.47461	17492.29688	17571.69141	17638.08398	17527.50977	17442.40039	17561.52734	17492.63281	17577.96875	17452.38867	17510.50781	17479.29492	17579.23828	17510.60156	17726.15234	17757.61328	18004.30664	17867.71289	17892.08984	17668.95313	17503.71094	17455.14648	17652.04492	17584.22656	17634.4082	17496.27539	17511.58789	17628.84375	17451.75391	17482.72852	17448.80664	17380.3418	17314.01953	16941.00977	16752.75	16451.42773	15955.27832	15537.80762	14975.84766	14553.67676	14080.69824	13838.50195	13714.57422	13421.56445	13069.82324
];
dem =reshape(dem,1,1152);
dem_max = max(dem);
dem = dem/dem_max;
dem = 2.834*dem;
plot(dem)