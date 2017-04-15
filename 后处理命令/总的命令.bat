md final
echo start resu
java -jar resulotion.jar Fscope\\F_predict_result.txt res\\combine.testFLabel.blank Fscope\\outPred_Fscope.txt

java -jar resulotion.jar Lscope\\L_predict_result.txt res\\combine.testLabel.blank Lscope\\outPred_Lscope.txt
echo start...combine
java -jar CombineProbAndResults.jar res\\F.res Fscope\\outPred_Fscope.txt Fscope\\outFResults.txt
java -jar CombineProbAndResults.jar res\\L.res Lscope\\outPred_Lscope.txt Lscope\\outLResults.txt
echo start...postprocess
java -jar GetPostProcessByProbs.jar  Fscope\\outFResults.txt Fscope\\postFscopeProcessbyProbs.txt
java -jar GetPostProcessByProbs.jar  Lscope\\outLResults.txt Lscope\\postLscopeProcessbyProbs.txt

echo start...combine
java -jar combineFandL.jar Fscope\\postFscopeProcessbyProbs.txt Lscope\\postLscopeProcessbyProbs.txt final\\outfinalProbs.txt


java  -jar evalScope.jar final\\outfinalProbs.txt finalSentencelevelFscore.txt


pause