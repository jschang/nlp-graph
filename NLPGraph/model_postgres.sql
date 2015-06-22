DROP TABLE input_channel;
DROP TABLE symbol;
DROP TABLE symbol_member;
DROP TABLE recollection;
DROP TABLE recollection_exception;

CREATE TABLE input_channel
(
  id bigserial NOT NULL,
  create_dtime timestamp without time zone DEFAULT now(),
  CONSTRAINT input_channel_pkey PRIMARY KEY (id)
)
WITH (
  OIDS=FALSE
);

CREATE TABLE symbol
(
  id bigserial NOT NULL,
  input_channel_id bigint,
  create_dtime timestamp without time zone DEFAULT now(),
  CONSTRAINT symbol_pkey PRIMARY KEY (id),
  CONSTRAINT symbol_input_channel_id_fk FOREIGN KEY (input_channel_id)
      REFERENCES input_channel (id) MATCH SIMPLE
      ON UPDATE NO ACTION ON DELETE NO ACTION
)
WITH (
  OIDS=FALSE
);

CREATE TABLE symbol_member
(
  symbol_id bigint,
  index integer NOT NULL,
  parent_symbol_id bigint NOT NULL,
  create_dtime timestamp without time zone DEFAULT now(),
  CONSTRAINT symbol_member_pkey PRIMARY KEY (parent_symbol_id, index),
  CONSTRAINT symbol_member_parent_symbol_id_fkey FOREIGN KEY (parent_symbol_id)
      REFERENCES symbol (id) MATCH SIMPLE
      ON UPDATE NO ACTION ON DELETE NO ACTION,
  CONSTRAINT symbol_member_symbol_id_fkey FOREIGN KEY (symbol_id)
      REFERENCES symbol (id) MATCH SIMPLE
      ON UPDATE NO ACTION ON DELETE NO ACTION
)
WITH (
  OIDS=FALSE
);

CREATE TABLE recollection
(
  id bigserial NOT NULL,
  symbol_id bigint,
  input_channel_id bigint,
  create_dtime timestamp without time zone,
  CONSTRAINT recollection_pkey PRIMARY KEY (id),
  CONSTRAINT recollection_input_channel_id_fkey FOREIGN KEY (input_channel_id)
      REFERENCES input_channel (id) MATCH SIMPLE
      ON UPDATE NO ACTION ON DELETE NO ACTION,
  CONSTRAINT recollection_symbol_id_fkey FOREIGN KEY (symbol_id)
      REFERENCES symbol (id) MATCH SIMPLE
      ON UPDATE NO ACTION ON DELETE NO ACTION
)
WITH (
  OIDS=FALSE
);

CREATE TABLE recollection_exception
(
  recollection_id bigint NOT NULL,
  operation_id smallint NOT NULL,
  member_index integer NOT NULL,
  symbol_id bigint,
  CONSTRAINT recollection_exception_pkey PRIMARY KEY (recollection_id, member_index),
  CONSTRAINT recollection_exception_symbol_id_fkey FOREIGN KEY (symbol_id)
      REFERENCES symbol (id) MATCH SIMPLE
      ON UPDATE NO ACTION ON DELETE NO ACTION
)
WITH (
  OIDS=FALSE
);