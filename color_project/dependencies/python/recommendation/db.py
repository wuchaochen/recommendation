# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import contextlib

from functools import wraps

from sqlalchemy import create_engine, Column, String, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker

SQL_ALCHEMY_DB_FILE = '/tmp/rec.db'
SQL_ALCHEMY_CONN = "sqlite:///" + SQL_ALCHEMY_DB_FILE
engine = None
Session = None


def prepare_db(user_engine=None, user_session=None, print_sql=False):
    global engine
    global Session
    if user_engine is not None and user_session is not None:
        engine = user_engine
        Session = user_session
    if engine is None or Session is None:
        engine_args = {'encoding': "utf-8"}
        if print_sql:
            engine_args['echo'] = True
        engine = create_engine(SQL_ALCHEMY_CONN, **engine_args)
        Session = scoped_session(
            sessionmaker(autocommit=False,
                         autoflush=False,
                         bind=engine,
                         expire_on_commit=False))


def clear_engine_and_session():
    global engine
    global Session
    engine = None
    Session = None


@contextlib.contextmanager
def create_session():
    prepare_db()
    session = Session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def provide_session(func):
    """
    Function decorator that provides a session if it isn't provided.
    If you want to reuse a session or run the function as part of a
    database transaction, you pass it to the function, if not this wrapper
    will create one and close it for you.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        arg_session = 'session'

        func_params = func.__code__.co_varnames
        session_in_args = arg_session in func_params and \
                          func_params.index(arg_session) < len(args)
        session_in_kwargs = arg_session in kwargs

        if session_in_kwargs or session_in_args:
            return func(*args, **kwargs)
        else:
            with create_session() as session:
                kwargs[arg_session] = session
                return func(*args, **kwargs)

    return wrapper


Base = declarative_base()


class User(Base):
    __tablename__ = "user"
    id = Column(Integer(), primary_key=True)
    uid = Column(Integer(), nullable=False)
    country = Column(Integer, nullable=False)


class UserClick(Base):
    __tablename__ = "user_click"
    id = Column(Integer(), primary_key=True)
    uid = Column(Integer(), nullable=False)
    fs_1 = Column(String(1024), nullable=False)
    fs_2 = Column(String(1024), nullable=False)


def init_db(uri=None):
    global SQL_ALCHEMY_CONN
    if uri is not None:
        SQL_ALCHEMY_CONN = uri
    prepare_db()
    Base.metadata.create_all(engine)


@provide_session
def get_user_info(uid, session=None):
    return session.query(User).filter(User.uid == uid).first()


@provide_session
def get_users_info(uids, session=None):
    return session.query(User).filter(User.uid.in_(uids)).all()


@provide_session
def get_user_click_info(uid, session=None):
    return session.query(UserClick).filter(UserClick.uid == uid).first()


@provide_session
def get_users_click_info(uids, session=None):
    return session.query(UserClick).filter(UserClick.uid.in_(uids)).all()


@provide_session
def update_user_click_info(uid, fs, session=None):
    user_click = session.query(UserClick).filter(UserClick.uid == uid).first()
    user_click.fs_2 = user_click.fs_1
    user_click.fs_1 = fs
    session.commit()


if __name__ == '__main__':
    from recommendation import config
    init_db(config.DbConn)
    res = get_users_click_info([2, 3, 6])
    for r in res:
        print(r.uid)