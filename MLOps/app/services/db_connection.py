import pandas as pd
import json
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, exc, text

# .env 파일에서 환경 변수 로드
load_dotenv()

class DatabaseService:
    """DB 커넥션 풀을 이용한 쿼리 서비스 클래스 (환경 변수 설정 및 타임아웃 적용)"""
    def __init__(self):
        """환경 변수에서 설정을 읽어와 커넥션 풀을 초기화합니다."""
        self.engine = None
        try:
            db_host = os.getenv('DB_HOST')
            db_port = os.getenv('DB_PORT')
            db_user = os.getenv('DB_USER')
            db_password = os.getenv('DB_PASSWORD')
            db_database = os.getenv('DB_DATABASE')

            if not all([db_host, db_port, db_user, db_password, db_database]):
                raise ValueError("DB 연결을 위한 모든 환경 변수가 설정되지 않았습니다.")

            # MySQL Connector/Python 용 SQLAlchemy URI
            db_uri = f"mysql+mysqlconnector://{db_user}:{db_password}@{db_host}:{db_port}/{db_database}"
            
            self.engine = create_engine(
                db_uri,
                pool_size=5,
                pool_recycle=3600, # 1시간마다 연결 재설정
                connect_args={'connect_timeout': 10}
            )
            print("SQLAlchemy 커넥션 풀 생성 성공")
        except (exc.SQLAlchemyError, ValueError) as e:
            print(f"커넥션 풀 생성 오류: {e}")

    def close_connection(self):
        if self.engine:
            self.engine.dispose()
            print("커넥션 풀 종료")

    def execute_query(self, query, params=None):
        """쿼리 실행 후 데이터프레임 반환"""
        if not self.engine:
            print("DB 엔진을 사용할 수 없습니다.")
            return None
        
        try:
            with self.engine.connect() as connection:
                df = pd.read_sql(text(query), connection, params=params)
                return df
        except exc.SQLAlchemyError as e:
            print(f"쿼리 실행 오류: {e}")
            return None

    def user_table_query(self):
        query = """
        SELECT * FROM user LIMIT 10;
        """
        return self.execute_query(query)

    def get_user_info_by_user_id(self, user_id):
        """user_id로 사용자 정보 조회"""
        query = f"""
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
            u.id = UNHEX(:user_id_hex)
        GROUP BY
            u.id, c.id;
        """
        user_id_hex = user_id.replace('-', '')
        df = self.execute_query(query, params={'user_id_hex': user_id_hex})
        if df is not None:
            df = df.rename(columns={
                'user_id': 'userid',
                'user_name': 'name',
                'tag_names': 'like_list'
            })
        return df