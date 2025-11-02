CREATE DATABASE  IF NOT EXISTS `salesdatabase` /*!40100 DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci */ /*!80016 DEFAULT ENCRYPTION='N' */;
USE `salesdatabase`;
-- MySQL dump 10.13  Distrib 8.0.32, for Win64 (x86_64)
--
-- Host: localhost    Database: salesdatabase
-- ------------------------------------------------------
-- Server version	8.0.32

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `customer`
--

DROP TABLE IF EXISTS `customer`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `customer` (
  `Id` int NOT NULL AUTO_INCREMENT,
  `CustomerID` int DEFAULT NULL,
  `Name` varchar(45) CHARACTER SET utf8mb3 COLLATE utf8mb3_unicode_ci DEFAULT NULL,
  `Gender` varchar(45) CHARACTER SET utf8mb3 COLLATE utf8mb3_unicode_ci DEFAULT NULL,
  `Age` int DEFAULT NULL,
  PRIMARY KEY (`Id`)
) ENGINE=InnoDB AUTO_INCREMENT=201 DEFAULT CHARSET=utf8mb3 COLLATE=utf8mb3_unicode_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `customer`
--

LOCK TABLES `customer` WRITE;
/*!40000 ALTER TABLE `customer` DISABLE KEYS */;
INSERT INTO `customer` VALUES (1,1,'lee Mean','Male',19),(2,2,'pinknguyen','Male',21),(3,3,'Lê Đảm','Female',20),(4,4,'Nguyễn Văn Toàn','Female',23),(5,5,'Trốc Gúi','Female',31),(6,6,'Kaio Nguyễn Hoàng','Female',22),(7,7,'quynhtp.cd','Female',35),(8,8,'jutjit','Female',23),(9,9,'dinh dang','Male',64),(10,10,'phuc05vietnam','Female',30),(11,11,'vuongtai','Male',67),(12,12,'datlee3010199','Female',35),(13,13,'hiepmts','Female',58),(14,14,'Quang Đạo Đồng','Female',24),(15,15,'PHẠM QUỐC TUẤN','Male',37),(16,16,'Thành Luân Trần','Male',22),(17,17,'Võ Nguyễn Thanh Loan','Female',35),(18,18,'Kim Trang Tran Thi','Male',20),(19,19,'Tuyết mùa hạ','Male',52),(20,20,'phamthithuhuong402','Female',35),(21,21,'Nguyễn Vạn Toàn','Male',35),(22,22,'Hồ Đồ','Male',25),(23,23,'Bich tiên','Female',46),(24,24,'Hoat Tran','Male',31),(25,25,'Đinh Đằng','Female',54),(26,26,'tanghongmong','Male',29),(27,27,'pha lê tiu','Female',45),(28,28,'Teo tèo','Male',35),(29,29,'Đinh Đáng','Female',40),(30,30,'btuankhanh','Female',23),(31,31,'Ty ty','Male',60),(32,32,'nhattaidnp','Female',21),(33,33,'meodethuongqn','Male',53),(34,34,'Phanbonhuuhai','Male',18),(35,35,'Huynh_lg','Female',49),(36,36,'ly son','Female',21),(37,37,'xuka_jp','Female',42),(38,38,'thinhpdfit','Female',30),(39,39,'Đinh Đang','Female',36),(40,40,'Nguyen Van Toan','Female',20),(41,41,'Nguyễn văn toản','Female',65),(42,42,'Tép Rô','Male',24),(43,43,'congluan99','Male',48),(44,44,'Nakamichivn','Female',31),(45,45,'dinhtd','Female',49),(46,46,'Lê Tấn Thắng','Female',24),(47,47,'Nhã Kỳ','Female',50),(48,48,'phu69sg','Female',27),(49,49,'Kiên Đào Trung','Female',29),(50,50,'hieuduong84','Female',31),(51,51,'hoang05081990','Female',49),(52,52,'Ice Tea','Male',33),(53,53,'kingofmine_88','Female',31),(54,54,'maibach66','Male',59),(55,55,'triệu quang vũ','Female',50),(56,56,'ân lê hồng','Male',47),(57,57,'Nguyễn Trai','Female',51),(58,58,'Trai Làng','Male',69),(59,59,'vau1vau2','Female',27),(60,60,'Ngọc Viettel Cáp Quang 0981116444','Male',53),(61,61,'Giang Lee','Male',70),(62,62,'khanhnc.nps','Male',19),(63,63,'hovy1995.nghean','Female',67),(64,64,'pe_tocngan91','Female',54),(65,65,'Nha Ky','Male',63),(66,66,'Trà Giang Nguyễn','Male',18),(67,67,'daohaidang07','Female',43),(68,68,'Linh Phuong','Female',68),(69,69,'cafelamson','Male',19),(70,70,'Hậu ACE','Female',32),(71,71,'gold170990','Male',70),(72,72,'Ductho103','Female',47),(73,73,'Hung Ton','Female',60),(74,74,'trannhiqt66','Female',60),(75,75,'Hong Kdoh','Male',59),(76,76,'caonhung','Male',26),(77,77,'phuotngao','Female',45),(78,78,'toanbeo','Male',40),(79,79,'hnt1969','Female',23),(80,80,'BonBon MyMy','Female',49),(81,81,'phuhai','Male',57),(82,82,'ngochec13','Male',38),(83,83,'Văn sỹ','Male',67),(84,84,'Nam Lu','Female',46),(85,85,'Phamhungviet1975','Female',21),(86,86,'tedicovn@gmail.com','Male',48),(87,87,'Lang Tu Mien Tay','Female',55),(88,88,'Hoàng Mã Vui','Female',22),(89,89,'nguyencongtri','Female',34),(90,90,'khanhthanhanh','Female',50),(91,91,'Lientan160891','Female',68),(92,92,'CNwanted .','Male',18),(93,93,'Hồ Việt Văn','Male',48),(94,94,'neverbackdown6789','Female',40),(95,95,'Ngô Minh Thuận','Female',32),(96,96,'Lamthu9286','Male',24),(97,97,'văng782','Female',47),(98,98,'Nicepet2007','Female',27),(99,99,'limichi9999','Male',48),(100,100,'Ngo Tung Minh','Male',20),(101,101,'Huutoan7426','Female',23),(102,102,'Sonhaudau','Female',49),(103,103,'vòng a vĩnh','Male',67),(104,104,'dinhthitho','Male',26),(105,105,'Kaoru4ever2000','Male',49),(106,106,'Ngọc Hà Trần','Female',21),(107,107,'Cố Quên Một Người','Female',66),(108,108,'hahgha','Male',54),(109,109,'HOANG-MT','Male',68),(110,110,'datphamshop','Male',66),(111,111,'nguyễn ngọc hội','Male',65),(112,112,'dunggiaodich','Female',19),(113,113,'nguyenthanhdat1802','Female',38),(114,114,'manhtrung86','Male',19),(115,115,'Triton','Female',18),(116,116,'Thắng Hương','Female',19),(117,117,'zuonghoi','Female',63),(118,118,'cần một ngôi nhà','Female',49),(119,119,'gaduaxe','Female',51),(120,120,'thailanfiditour','Female',50),(121,121,'Mạnh Phương','Male',27),(122,122,'trần duy khiêm','Female',38),(123,123,'Sac Viet ','Female',40),(124,124,'dinhdam_87','Male',39),(125,125,'hungchu1','Female',23),(126,126,'Nharatngheodq','Female',31),(127,127,'Phuongtd','Male',43),(128,128,'Cong Vu','Male',40),(129,129,'Minh Huy Ho','Male',59),(130,130,'ngoquangtrung.haui','Male',38),(131,131,'hoanho23.1','Male',47),(132,132,'tranvuhai2005','Male',39),(133,133,'hungtn982 hoang','Female',25),(134,134,'phạm chí dũng','Female',31),(135,135,'lenguyenccts','Male',20),(136,136,'Thiên Nam','Female',29),(137,137,'Tran Duy Khiem','Female',44),(138,138,'công vũ','Male',32),(139,139,'lnndgk','Male',19),(140,140,'tyty281207','Female',35),(141,141,'Thạch LD','Female',57),(142,142,'Bao Vo','Male',32),(143,143,'quyenhotan','Female',28),(144,144,'thuyanh','Female',32),(145,145,'Trần Hùng','Male',25),(146,146,'trần hung','Male',28),(147,147,'nguyendinhsơn','Male',48),(148,148,'Tran hung','Female',32),(149,149,'dung do thanh','Female',34),(150,150,'Nhovehanoithu','Male',34),(151,151,'banjvancan007','Male',43),(152,152,'Mạnh Ninh','Male',39),(153,153,'Nhat HD','Female',44),(154,154,'lahanqua','Female',38),(155,155,'Thư Trương','Female',47),(156,156,'ngohang62','Female',27),(157,157,'LETHINHULAN.BVBR','Male',37),(158,158,'Ths.Kts Phạm Thanh Truyền','Female',30),(159,159,'ngmthang','Male',34),(160,160,'Happy day','Female',30),(161,161,'Minhkhang2015','Female',56),(162,162,'Mai Nha Hong','Female',29),(163,163,'nguyenbang12','Male',19),(164,164,'Ðịch Tiến Ta Tui','Female',31),(165,165,'Trunghovan861','Male',50),(166,166,'phanminhnhat','Female',36),(167,167,'hinsky.huy','Male',42),(168,168,'Linh phan nhat','Female',33),(169,169,'Ly Thai','Female',36),(170,170,'toilatoi1','Male',32),(171,171,'xuan quyet vu','Male',40),(172,172,'ThuýAnh','Male',28),(173,173,'tranvanhoa10061966','Male',36),(174,174,'Trần Hưng','Male',36),(175,175,'Thu Truong','Female',52),(176,176,'Không Biết','Female',30),(177,177,'trân hùng','Male',58),(178,178,'Mr. EG','Male',27),(179,179,'danviet1946','Male',59),(180,180,'Vũ huy Đại','Male',35),(181,181,'No Seven','Female',37),(182,182,'Ngoc Cuong Duong','Female',32),(183,183,'vanchung.cao','Male',46),(184,184,'Xe đạp','Female',29),(185,185,'CÔNG SƠN ĐINH','Female',41),(186,186,'dqdung57','Male',30),(187,187,'Đức Luân','Female',54),(188,188,'chauphu902gmail.com','Male',28),(189,189,'XE DAP','Female',41),(190,190,'Kmacblad Trungpt','Female',36),(191,191,'Bảo Net','Female',34),(192,192,'khfdmnj2367','Female',32),(193,193,'Phan Xuân Tĩnh','Male',33),(194,194,'duytien','Female',38),(195,195,'Nguyễn Tin','Female',47),(196,196,'Cytheria Tran','Female',35),(197,197,'truongbo37','Female',45),(198,198,'vohoan2408','Male',32),(199,199,'vinhdq.vn','Male',32),(200,200,'Nha6942','Male',30);
/*!40000 ALTER TABLE `customer` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2025-09-04 16:40:24
