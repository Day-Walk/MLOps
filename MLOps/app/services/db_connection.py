import mysql.connector
from mysql.connector import pooling, Error
import pandas as pd
import json
import os

class DatabaseService:
    """DB 커넥션 풀을 이용한 쿼리 서비스 클래스 (환경 변수 설정 및 타임아웃 적용)"""
    def __init__(self):
        """환경 변수에서 설정을 읽어와 커넥션 풀을 초기화합니다."""
        self.pool = None
        try:
            db_config = {
                'host': os.getenv('DB_HOST', '15.164.50.188'),
                'port': int(os.getenv('DB_PORT', 3307)),
                'user': os.getenv('DB_USER', 'root'),
                'password': os.getenv('DB_PASSWORD', 'pwd1234'),
                'database': os.getenv('DB_DATABASE', 'daywalk')
            }
            pool_size = int(os.getenv('DB_POOL_SIZE', 5))

            self.pool = mysql.connector.pooling.MySQLConnectionPool(
                pool_name="daywalk_pool",
                pool_size=pool_size,
                pool_reset_session=True,
                **db_config
            )
            print("MySQL 커넥션 풀 생성 성공")
        except Error as e:
            print(f"커넥션 풀 생성 오류: {e}")

    def _get_connection(self, timeout=3):
        """풀에서 커넥션을 가져옵니다. 타임아웃을 적용하여 무한 대기를 방지합니다."""
        if not self.pool:
            print("커넥션 풀을 사용할 수 없습니다.")
            return None
        try:
            # 타임아웃(초)을 설정하여 커넥션을 기다립니다.
            return self.pool.get_connection(timeout=timeout)
        except pooling.PoolError as e:
            print(f"풀에서 커넥션을 가져오는 데 실패했습니다 (타임아웃 또는 풀 문제): {e}")
            return None
        except Error as e:
            print(f"커넥션 가져오는 중 알 수 없는 오류 발생: {e}")
            return None

    def execute_query(self, query, params=None):
        """쿼리 실행 후 데이터프레임 반환"""
        connection = self._get_connection()
        if not connection:
            return None
        
        try:
            df = pd.read_sql(query, connection, params=params)
            return df
        except Error as e:
            print(f"쿼리 실행 오류: {e}")
            return None
        finally:
            if connection and connection.is_connected():
                connection.close()
                print("사용한 커넥션을 풀에 반환했습니다.")

    def get_user_info_by_user_id(self, user_id):
        """user_id로 사용자 정보 조회"""
        query = """
        SELECT
            HEX(u.id) AS user_id,
            u.name AS user_name,
            u.gender,
            u.age,
            c.name AS category_name,
            IFNULL(JSON_ARRAYAGG(t.keyword), JSON_ARRAY()) AS tag_names
        FROM
            user u
        LEFT JOIN user_like ul ON u.id = ul.user_id
        LEFT JOIN category c ON ul.category_id = c.id
        LEFT JOIN JSON_TABLE(
            ul.tag_list,
            '$[*]' COLUMNS (tag_id VARCHAR(36) PATH '$')
        ) AS jt ON TRUE
        LEFT JOIN tag t ON t.id = UNHEX(REPLACE(jt.tag_id, '-', ''))
        WHERE
            u.id = UNHEX(%s)
        GROUP BY
            u.id, c.id;
        """
        user_id_hex = user_id[2:] if user_id.startswith('0x') else user_id
        return self.execute_query(query, params=(user_id_hex,))

if __name__ == '__main__':
    # 아래 코드는 웹 프레임워크(예: FastAPI)의 시작 지점에서 한 번만 실행되어야 합니다.
    # export DB_HOST=... 와 같은 방식으로 환경 변수 설정 후 실행할 수 있습니다.
    db_service = DatabaseService()
    
    if db_service.pool:
        # 특정 사용자 데이터 조회 예시
        test_user_id = '0x0034B410791D47A38ABFE03E0898A61A' 
        user_data_df = db_service.get_user_info_by_user_id(test_user_id)
        
        if user_data_df is not None:
            print(f"{test_user_id} 사용자의 전체 데이터를 성공적으로 가져왔습니다.")
            print(user_data_df.to_string())
