전부 zip으로 다운 받은 후 , 압축 풀기 이후

0. 깃허브에 있는 final_backup.sql 을 사용해 coursemate DB 생성
  
1. backend에서 npm install이랑 .env 자신 DB, 환경설정에 맞춰서 변경

2. Front/cosmanager, cosuser 각각 들어가서 npm install 

3. cosmanager/src/api/client.js 에서 자신 설정에 맞게 주소 바꾸기 + cosuser/api/client.js 에서도 동일하게 변경

4. cosuser/components/PlaceCard에서 주소 똑같이 변경

5. Ai/models 의 download_guide를 통해 동일한 위치에 course_mate_model.pt 넣기
